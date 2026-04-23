"""Assignment 03 Task 1: Bundle Adjustment implemented with PyTorch.

This script optimizes:
1) Shared focal length f
2) Per-view camera extrinsics (Euler angles + translation)
3) 3D coordinates of all points

Outputs:
- results/loss_curve.png
- results/reconstruction.obj  (colored point cloud)
- results/ba_params.npz
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles (XYZ convention) to rotation matrices.

    Args:
        euler: (..., 3) tensor [rx, ry, rz] in radians.
    Returns:
        (..., 3, 3) rotation matrix.
    """
    rx, ry, rz = euler.unbind(dim=-1)
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    # Rx
    ones = torch.ones_like(rx)
    zeros = torch.zeros_like(rx)
    rx_mat = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cx, -sx], dim=-1),
            torch.stack([zeros, sx, cx], dim=-1),
        ],
        dim=-2,
    )

    # Ry
    ry_mat = torch.stack(
        [
            torch.stack([cy, zeros, sy], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sy, zeros, cy], dim=-1),
        ],
        dim=-2,
    )

    # Rz
    rz_mat = torch.stack(
        [
            torch.stack([cz, -sz, zeros], dim=-1),
            torch.stack([sz, cz, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )

    # Convention XYZ: rotate around X, then Y, then Z
    return rx_mat @ ry_mat @ rz_mat


def project_points(
    points_3d: torch.Tensor,
    euler_angles: torch.Tensor,
    translations: torch.Tensor,
    focal: torch.Tensor,
    cx: float,
    cy: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project 3D points to 2D for a set of views.

    Args:
        points_3d: (P, 3)
        euler_angles: (B, 3)
        translations: (B, 3)
        focal: scalar tensor
        cx, cy: principal point
    Returns:
        pred_uv: (B, P, 2)
        cam_xyz: (B, P, 3)
    """
    rot = euler_xyz_to_matrix(euler_angles)  # (B, 3, 3)
    cam_xyz = torch.einsum("bij,pj->bpi", rot, points_3d) + translations[:, None, :]
    z = cam_xyz[..., 2]
    z_safe = torch.where(z.abs() < 1e-4, z.sign() * 1e-4, z)

    # Assignment convention:
    # u = -f * Xc/Zc + cx
    # v =  f * Yc/Zc + cy
    u = -focal * cam_xyz[..., 0] / z_safe + cx
    v = focal * cam_xyz[..., 1] / z_safe + cy
    pred_uv = torch.stack([u, v], dim=-1)
    return pred_uv, cam_xyz


def save_colored_obj(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Save colored point cloud as OBJ: each line is `v x y z r g b`."""
    with path.open("w", encoding="utf-8") as f:
        for p, c in zip(xyz, rgb):
            f.write(
                f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundle Adjustment with PyTorch")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for logs and reconstruction",
    )
    parser.add_argument("--iters", type=int, default=2000, help="Optimization steps")
    parser.add_argument("--lr", type=float, default=8e-3, help="Learning rate")
    parser.add_argument(
        "--view_batch",
        type=int,
        default=10,
        help="Number of views sampled per iteration",
    )
    parser.add_argument(
        "--point_batch",
        type=int,
        default=6000,
        help="Number of points sampled per iteration",
    )
    parser.add_argument(
        "--init_distance",
        type=float,
        default=2.5,
        help="Initialize translation z as -init_distance",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_every", type=int, default=50, help="Print interval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points2d_npz = np.load(data_dir / "points2d.npz")
    keys = sorted(points2d_npz.files)
    obs = np.stack([points2d_npz[k] for k in keys], axis=0).astype(np.float32)  # (V,N,3)
    obs_xy = obs[..., :2]
    vis = obs[..., 2]

    colors = np.load(data_dir / "points3d_colors.npy").astype(np.float32)
    colors = np.clip(colors, 0.0, 1.0)

    n_views, n_points, _ = obs.shape
    img_w = 1024.0
    img_h = 1024.0
    cx = img_w / 2.0
    cy = img_h / 2.0

    # Typical FoV 60 deg -> f ~ H / (2 * tan(FoV/2))
    init_f = img_h / (2.0 * math.tan(math.radians(60.0) / 2.0))

    device = torch.device("cpu")
    obs_xy_t = torch.from_numpy(obs_xy).to(device)
    vis_t = torch.from_numpy(vis).to(device)

    # Learnable variables
    points_3d = torch.nn.Parameter(0.1 * torch.randn(n_points, 3, device=device))
    euler_angles = torch.nn.Parameter(torch.zeros(n_views, 3, device=device))

    t_init = torch.tensor([0.0, 0.0, -args.init_distance], device=device).repeat(n_views, 1)
    translations = torch.nn.Parameter(t_init)

    # Positive focal via exponential parameterization.
    focal_log = torch.nn.Parameter(torch.tensor(math.log(init_f), device=device))

    optimizer = torch.optim.Adam(
        [points_3d, euler_angles, translations, focal_log],
        lr=args.lr,
    )

    losses: list[float] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] = {}

    for step in range(1, args.iters + 1):
        view_idx = torch.randperm(n_views, device=device)[: min(args.view_batch, n_views)]
        point_idx = torch.randperm(n_points, device=device)[: min(args.point_batch, n_points)]

        p = points_3d[point_idx]
        e = euler_angles[view_idx]
        t = translations[view_idx]
        focal = torch.exp(focal_log)

        pred_uv, cam_xyz = project_points(p, e, t, focal, cx, cy)  # (B,P,2), (B,P,3)
        target_uv = obs_xy_t[view_idx][:, point_idx]
        mask = vis_t[view_idx][:, point_idx]

        diff = pred_uv - target_uv
        reproj = torch.sum(diff * diff, dim=-1)  # (B,P)
        data_loss = (reproj * mask).sum() / mask.sum().clamp_min(1.0)

        # Keep points roughly centered to stabilize gauge freedom.
        center_reg = points_3d.mean(dim=0).pow(2).sum()
        # Keep camera in front of object (Zc should be negative under this setup).
        z_front_penalty = torch.relu(cam_xyz[..., 2] + 1e-2).mean()
        # Prevent exploding Euler angles / translations.
        pose_reg = 1e-4 * (euler_angles.pow(2).mean() + translations.pow(2).mean())

        loss = data_loss + 1e-3 * center_reg + 1e-2 * z_front_penalty + pose_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = float(data_loss.detach().cpu().item())
        losses.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {
                "points_3d": points_3d.detach().cpu().clone(),
                "euler_angles": euler_angles.detach().cpu().clone(),
                "translations": translations.detach().cpu().clone(),
                "focal": torch.exp(focal_log).detach().cpu().clone(),
            }

        if step == 1 or step % args.log_every == 0 or step == args.iters:
            print(
                f"[{step:04d}/{args.iters}] "
                f"reproj_mse={loss_val:.4f} "
                f"f={float(torch.exp(focal_log).item()):.3f}"
            )

    # Save curve
    plt.figure(figsize=(8, 4))
    plt.plot(losses, lw=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Reprojection MSE (visible points)")
    plt.title("Bundle Adjustment Optimization Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=160)
    plt.close()

    # Save best reconstruction + params
    xyz_best = best_state["points_3d"].numpy()
    save_colored_obj(out_dir / "reconstruction.obj", xyz_best, colors)

    np.savez(
        out_dir / "ba_params.npz",
        focal=np.array([float(best_state["focal"].item())], dtype=np.float32),
        euler_angles=best_state["euler_angles"].numpy().astype(np.float32),
        translations=best_state["translations"].numpy().astype(np.float32),
        points3d=xyz_best.astype(np.float32),
    )

    print("\nOptimization finished.")
    print(f"Best reprojection MSE: {best_loss:.6f}")
    print(f"Saved: {out_dir / 'loss_curve.png'}")
    print(f"Saved: {out_dir / 'reconstruction.obj'}")
    print(f"Saved: {out_dir / 'ba_params.npz'}")


if __name__ == "__main__":
    main()
