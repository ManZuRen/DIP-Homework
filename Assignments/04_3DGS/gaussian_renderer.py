import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np
import cv2


class GaussianRenderer(nn.Module):
    def __init__(self, image_height: int, image_width: int, render_chunk_size: int = 256):
        super().__init__()
        self.H = image_height
        self.W = image_width
        self.render_chunk_size = render_chunk_size
        
        # Pre-compute pixel coordinates grid
        y, x = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing='ij'
        )
        # Shape: (H, W, 2)
        self.register_buffer('pixels', torch.stack([x, y], dim=-1))


    def compute_projection(
        self,
        means3D: torch.Tensor,          # (N, 3)
        covs3d: torch.Tensor,           # (N, 3, 3)
        K: torch.Tensor,                # (3, 3)
        R: torch.Tensor,                # (3, 3)
        t: torch.Tensor                 # (3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = means3D.shape[0]
        
        # 1. Transform points to camera space
        cam_points = means3D @ R.T + t.unsqueeze(0) # (N, 3)
        
        # 2. Get depths before projection for proper sorting and clipping
        depths = cam_points[:, 2]  # (N, )
        
        # 3. Project to screen space using camera intrinsics
        z_safe = depths.clamp(min=1e-6)
        means2D = torch.stack([
            K[0, 0] * cam_points[:, 0] / z_safe + K[0, 2],
            K[1, 1] * cam_points[:, 1] / z_safe + K[1, 2],
        ], dim=-1) # (N, 2)
        
        # 4. Transform covariance to camera space and then to 2D
        # Compute Jacobian of perspective projection
        J_proj = torch.zeros((N, 2, 3), device=means3D.device, dtype=means3D.dtype)
        fx, fy = K[0, 0], K[1, 1]
        x, y, z = cam_points.unbind(dim=-1)
        z = z.clamp(min=1e-6)
        z2 = z * z
        J_proj[:, 0, 0] = fx / z
        J_proj[:, 0, 2] = -fx * x / z2
        J_proj[:, 1, 1] = fy / z
        J_proj[:, 1, 2] = -fy * y / z2
        
        # Transform covariance to camera space
        R_batch = R.unsqueeze(0).expand(N, -1, -1)
        covs_cam = torch.bmm(R_batch, torch.bmm(covs3d, R_batch.permute(0, 2, 1)))  # (N, 3, 3)
        
        # Project to 2D
        covs2D = torch.bmm(J_proj, torch.bmm(covs_cam, J_proj.permute(0, 2, 1)))  # (N, 2, 2)
        covs2D = 0.5 * (covs2D + covs2D.permute(0, 2, 1))
        
        return means2D, covs2D, depths

    def compute_gaussian_values(
        self,
        means2D: torch.Tensor,    # (N, 2)
        covs2D: torch.Tensor,     # (N, 2, 2)
        pixels: torch.Tensor      # (H, W, 2)
    ) -> torch.Tensor:           # (N, H, W)
        N = means2D.shape[0]
        H, W = pixels.shape[:2]
        
        # Compute offset from mean (N, H, W, 2)
        dx = pixels.unsqueeze(0) - means2D.reshape(N, 1, 1, 2)
        
        # Add a small low-pass term to keep projected covariances invertible.
        eps = 1e-4
        covs2D = covs2D + eps * torch.eye(2, device=covs2D.device, dtype=covs2D.dtype).unsqueeze(0)
        
        # Compute 2x2 inverse explicitly to avoid expensive batched matrix inverse.
        a = covs2D[:, 0, 0]
        b = covs2D[:, 0, 1]
        c = covs2D[:, 1, 1]
        det = (a * c - b * b).clamp(min=1e-8)
        inv00 = c / det
        inv01 = -b / det
        inv11 = a / det
        dx0 = dx[..., 0]
        dx1 = dx[..., 1]
        exponent = -0.5 * (
            inv00.view(N, 1, 1) * dx0 * dx0
            + 2.0 * inv01.view(N, 1, 1) * dx0 * dx1
            + inv11.view(N, 1, 1) * dx1 * dx1
        )
        norm = 1.0 / (2.0 * torch.pi * torch.sqrt(det))
        gaussian = norm.view(N, 1, 1) * torch.exp(exponent.clamp(min=-80.0, max=0.0)) ## (N, H, W)
        gaussian = torch.nan_to_num(gaussian, nan=0.0, posinf=0.0, neginf=0.0)
    
        return gaussian

    def _render_chunk(
        self,
        means2D: torch.Tensor,
        covs2D: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        pixels: torch.Tensor,
        transmittance: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gaussian_values = self.compute_gaussian_values(means2D, covs2D, pixels)  # (M, H, W)
        alphas = opacities.view(-1, 1, 1) * gaussian_values
        alphas = alphas.clamp(min=0.0, max=0.999)
        local_trans = torch.cumprod(
            torch.cat(
                [torch.ones((1, self.H, self.W), device=alphas.device, dtype=alphas.dtype),
                 1.0 - alphas + 1e-10],
                dim=0
            ),
            dim=0
        )[:-1]
        weights = alphas * local_trans * transmittance.unsqueeze(0)
        rendered_delta = (weights.unsqueeze(-1) * colors.view(-1, 1, 1, 3)).sum(dim=0)
        next_transmittance = transmittance * torch.prod(1.0 - alphas + 1e-10, dim=0)
        return rendered_delta, next_transmittance

    def forward(
            self,
            means3D: torch.Tensor,          # (N, 3)
            covs3d: torch.Tensor,           # (N, 3, 3)
            colors: torch.Tensor,           # (N, 3)
            opacities: torch.Tensor,        # (N, 1)
            K: torch.Tensor,                # (3, 3)
            R: torch.Tensor,                # (3, 3)
            t: torch.Tensor                 # (3, 1)
    ) -> torch.Tensor:
        # 1. Project to 2D, means2D: (N, 2), covs2D: (N, 2, 2), depths: (N,)
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)
        
        # 2. Frustum and screen-space mask. Keep a conservative covariance
        # radius so splats just outside the image can still contribute.
        radius = 3.0 * torch.sqrt(
            torch.clamp(torch.maximum(covs2D[:, 0, 0], covs2D[:, 1, 1]), min=1e-6)
        )
        radius = radius.clamp(min=1.0, max=float(max(self.H, self.W)))
        finite_mask = (
            torch.isfinite(means2D).all(dim=-1)
            & torch.isfinite(covs2D).flatten(1).all(dim=-1)
            & torch.isfinite(depths)
        )
        valid_mask = (
            finite_mask
            & (depths > 1.0)
            & (depths < 50.0)
            & (means2D[:, 0] >= -radius)
            & (means2D[:, 0] <= self.W - 1 + radius)
            & (means2D[:, 1] >= -radius)
            & (means2D[:, 1] <= self.H - 1 + radius)
        )
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            return torch.zeros((self.H, self.W, 3), device=means3D.device, dtype=means3D.dtype)
        
        # 3. Sort visible gaussians by depth.
        indices = valid_indices[torch.argsort(depths[valid_indices], dim=0, descending=False)]
        means2D = means2D[indices]      # (N, 2)
        covs2D = covs2D[indices]       # (N, 2, 2)
        colors = colors[ indices]       # (N, 3)
        opacities = opacities[indices] # (N, 1)
        N = means2D.shape[0]

        # 4-8. Chunked Gaussian evaluation and alpha composition. This keeps
        # peak memory bounded by render_chunk_size * H * W instead of N * H * W.
        rendered = torch.zeros((self.H, self.W, 3), device=means3D.device, dtype=means3D.dtype)
        transmittance = torch.ones((self.H, self.W), device=means3D.device, dtype=means3D.dtype)
        use_checkpoint = torch.is_grad_enabled() and (
            means2D.requires_grad or covs2D.requires_grad or colors.requires_grad or opacities.requires_grad
        )
        for start in range(0, N, self.render_chunk_size):
            end = min(start + self.render_chunk_size, N)
            chunk_args = (
                means2D[start:end],
                covs2D[start:end],
                colors[start:end],
                opacities[start:end],
                self.pixels,
                transmittance,
            )
            if use_checkpoint:
                rendered_delta, transmittance = checkpoint(
                    self._render_chunk, *chunk_args, use_reentrant=False
                )
            else:
                rendered_delta, transmittance = self._render_chunk(*chunk_args)
            rendered = rendered + rendered_delta
        
        return rendered
