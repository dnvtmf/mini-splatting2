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
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..')))
import time
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from fused_ssim import fused_ssim
from gaussian_renderer import render, render_imp, render_simp, render_depth
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, read_config

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, args):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset.is_blender)
    deform.train_setting(opt)

    scene = Scene(dataset, gaussians, resolution_scales=[1, 2])
    gaussians.training_setup(opt, use_adam=False)
    gaussians.init_culling(len(scene.getTrainCameras()))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
    for iteration in range(1, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        deform.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras_warn_up(
                iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()
            # viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        fid = viewpoint_cam.fid

        N = gaussians.get_xyz.shape[0]
        time_input = fid.unsqueeze(0).expand(N, -1)

        ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(
            N, - 1) * time_interval * smooth_term(iteration)
        d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)
        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = d_xyz.detach(), d_rotation.detach(), d_scaling.detach()
        # Render
        render_pkg = render_imp(viewpoint_cam, gaussians, pipe, background, d_xyz=d_xyz, d_rotation=d_rotation,
                                d_scaling=d_scaling, culling=gaussians._culling[:, viewpoint_cam.uid])
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
            "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # depth = render_pkg["depth"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))  # ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "num": f"{gaussians._xyz.shape[0]:.2e}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                if gaussians._culling[:, viewpoint_cam.uid].sum() == 0:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                else:
                    # normalize xy gradient after culling
                    gaussians.add_densification_stats_culling(
                        viewspace_point_tensor, visibility_filter, gaussians.factor_culling)

                area_max = render_pkg["area_max"]
                mask_blur = torch.logical_or(mask_blur, area_max > (image.shape[1] * image.shape[2] / 5000))

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and \
                    iteration != args.depth_reinit_iter:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune_mask(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                     size_threshold, mask_blur)
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                    # print(f'after densify_and_prune_mask {gaussians._xyz.shape[0]} Gaussians')
                # if iteration % opt.opacity_reset_interval == 0 or (
                #     dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()
            if iteration == args.depth_reinit_iter:
                num_depth = gaussians._xyz.shape[0] * args.num_depth_factor

                # intersection_preserving for better point cloud reconstruction result at the early stage,
                # not affect rendering quality
                gaussians.intersection_preserving(scene, render_simp, iteration, args, pipe, background, deform=deform)
                pts, rgb = gaussians.depth_reinit(scene, render_depth, iteration, num_depth, args, pipe, background,
                                                  deform=None)

                gaussians.reinitial_pts(pts, rgb)

                gaussians.training_setup(opt)
                gaussians.init_culling(len(scene.getTrainCameras()))
                mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                torch.cuda.empty_cache()
                # print(f'after depth_reinit {gaussians._xyz.shape[0]} Gaussians')
                # print(gaussians._xyz.shape)

            if iteration >= args.aggressive_clone_from_iter and iteration % args.aggressive_clone_interval == 0 and iteration != args.depth_reinit_iter:
                gaussians.culling_with_clone(scene, render_simp, iteration, args, pipe, background, deform=deform)
                torch.cuda.empty_cache()
                mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                # print(f'after culling_with_clone {gaussians._xyz.shape[0]} Gaussians')
                # print(gaussians._xyz.shape)

            if iteration == args.simp_iteration1:
                gaussians.culling_with_intersection_sampling(
                    scene, render_simp, iteration, args, pipe, background, deform=deform
                )
                gaussians.max_sh_degree = dataset.sh_degree
                gaussians.extend_features_rest()

                gaussians.training_setup(opt)
                torch.cuda.empty_cache()
                # print(gaussians._xyz.shape)

            if iteration == args.simp_iteration2:
                gaussians.culling_with_intersection_preserving(
                    scene, render_simp, iteration, args, pipe, background, deform=deform
                )
                torch.cuda.empty_cache()
                # print(gaussians._xyz.shape)

            if iteration == (args.simp_iteration2 + opt.iterations) // 2:
                gaussians.init_culling(len(scene.getTrainCameras()))

            # Optimizer step
            if iteration < opt.iterations:
                visible = radii > 0
                gaussians.optimizer.step(visible, radii.shape[0])
                gaussians.optimizer.zero_grad(set_to_none=True)

                deform.optimizer.step()
                deform.optimizer.zero_grad()

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz=d_xyz, d_rotation=d_rotation,
                                   d_scaling=d_scaling)[
                            "render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--imp_metric", type=str, default='indoor')

    parser.add_argument("--config_path", type=str)

    parser.add_argument("--aggressive_clone_from_iter", type=int, default=500)
    parser.add_argument("--aggressive_clone_interval", type=int, default=250)

    parser.add_argument("--warn_until_iter", type=int, default=3_000)
    parser.add_argument("--depth_reinit_iter", type=int, default=2_000)
    parser.add_argument("--num_depth_factor", type=float, default=1)

    parser.add_argument("--simp_iteration1", type=int, default=3_000)
    parser.add_argument("--simp_iteration2", type=int, default=8_000)
    parser.add_argument("--sampling_factor", type=float, default=0.6)

    args = parser.parse_args(sys.argv[1:])
    args = read_config(parser)
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    torch.cuda.synchronize()
    time_start = time.time()
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args)
    torch.cuda.synchronize()
    time_end = time.time()
    time_total = time_end - time_start
    print(f'time: {time_total:.4f}s')

    # All done
    print("\nTraining complete.")
