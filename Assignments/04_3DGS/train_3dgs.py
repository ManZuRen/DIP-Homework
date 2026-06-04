#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import time
import json
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import Image, ImageDraw
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def tensor_to_uint8_image(image):
    image = torch.clamp(image.detach(), 0.0, 1.0)
    image = (image * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
    return Image.fromarray(image)

def save_debug_comparison(scene, pipe, background, iteration, debug_views):
    debug_dir = os.path.join(scene.model_path, "debug_images")
    os.makedirs(debug_dir, exist_ok=True)

    cameras = scene.getTrainCameras()
    selected = cameras[:min(debug_views, len(cameras))]
    cells = []
    with torch.no_grad():
        for viewpoint in selected:
            rendered = render(viewpoint, scene.gaussians, pipe, background)["render"]
            gt = viewpoint.original_image.to("cuda")
            render_img = tensor_to_uint8_image(rendered)
            gt_img = tensor_to_uint8_image(gt)
            cells.append((gt_img, render_img, viewpoint.image_name))

    if not cells:
        return

    cell_w, cell_h = cells[0][0].size
    label_h = 24
    grid = Image.new("RGB", (cell_w * len(cells), cell_h * 2 + label_h), (0, 0, 0))
    draw = ImageDraw.Draw(grid)
    draw.text((6, 4), f"iteration {iteration} | top: GT, bottom: Render", fill=(255, 255, 0))
    for i, (gt_img, render_img, name) in enumerate(cells):
        x = i * cell_w
        grid.paste(gt_img, (x, label_h))
        grid.paste(render_img, (x, label_h + cell_h))
        draw.text((x + 6, label_h + 6), name, fill=(0, 255, 0))

    grid.save(os.path.join(debug_dir, f"iter_{iteration:06d}.png"))

def append_metrics(model_path, payload):
    metrics_path = os.path.join(model_path, "metrics.jsonl")
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, debug_save_interval, debug_views):
    # 初始化迭代次数
    first_iter = 0
    # 准备输出和日志记录器
    tb_writer = prepare_output_and_logger(dataset)
    # 初始化高斯模型
    gaussians = GaussianModel(dataset.sh_degree)
    # 初始化场景
    scene = Scene(dataset, gaussians)
    # 设置高斯模型的训练参数
    gaussians.training_setup(opt)
    # 如果有检查点，则加载检查点
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 设置背景颜色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    train_start_time = time.time()

    # 初始化迭代开始和结束事件
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # 初始化视角堆栈
    viewpoint_stack = None
    # 初始化损失值
    ema_loss_for_log = 0.0
    # 初始化进度条
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # 迭代训练
    for iteration in range(first_iter, opt.iterations + 1):        
        # 如果网络GUI连接为空，则尝试连接
        if network_gui.conn == None:
            network_gui.try_connect()
        # 如果网络GUI连接不为空，则接收数据
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                # 如果有自定义相机，则渲染图像
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                # 发送图像
                network_gui.send(net_image_bytes, dataset.source_path)
                # 如果需要训练，则跳出循环
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # 记录迭代开始时间
        iter_start.record()

        # 更新学习率
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if debug_save_interval > 0 and (iteration % debug_save_interval == 0 or iteration == opt.iterations):
                save_debug_comparison(scene, pipe, background, iteration, debug_views)
                elapsed_sec = time.time() - train_start_time
                metrics = {
                    "iteration": iteration,
                    "loss": float(loss.item()),
                    "l1_loss": float(Ll1.item()),
                    "iter_time_ms": float(iter_start.elapsed_time(iter_end)),
                    "elapsed_sec": float(elapsed_sec),
                    "gaussians": int(gaussians.get_xyz.shape[0]),
                    "peak_memory_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024),
                }
                append_metrics(scene.model_path, metrics)
                print("\n[ITER {}] Debug saved | loss {:.6f} | gaussians {} | peak_mem {:.1f} MB | elapsed {:.1f}s".format(
                    iteration, metrics["loss"], metrics["gaussians"], metrics["peak_memory_mb"], metrics["elapsed_sec"]
                ))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    print("gaussian",gaussians._xyz.shape[0])
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    summary = {
        "total_iterations": int(opt.iterations),
        "total_time_sec": float(time.time() - train_start_time),
        "final_gaussians": int(gaussians.get_xyz.shape[0]),
        "peak_memory_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024),
        "model_path": scene.model_path,
    }
    with open(os.path.join(scene.model_path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

def prepare_output_and_logger(args):    
    # 如果没有指定模型路径，则生成一个唯一的字符串作为模型路径
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--debug_save_interval", type=int, default=1000,
                        help="Save GT/render debug comparison every N iterations. Set <=0 to disable.")
    parser.add_argument("--debug_views", type=int, default=4,
                        help="Number of training views included in each debug comparison image.")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.checkpoint_iterations is not None and args.iterations not in args.checkpoint_iterations:
        args.checkpoint_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.debug_save_interval, args.debug_views)

    # All done
    print("\nTraining complete.")
