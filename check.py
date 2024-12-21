import os
from random import randint
import sys
import uuid
from argparse import ArgumentParser, Namespace
import time
from pathlib import Path

import torch
from tqdm import tqdm

from extension.utils.test_utils import get_rel_error
from utils.loss_utils import l1_loss, ssim
from fused_ssim import fused_ssim
from gaussian_renderer import network_gui
from gaussian_renderer import render, render_imp, render_simp, render_depth
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams, read_config
from lpipsPyTorch import lpips
import extension as ext
from extension import utils
from NeRF.networks.gaussian_splatting.mini_splatting_2 import MiniSplatting2
from NeRF.datasets.colmap_dataset import ColmapDataset, NERF_DATASETS
from diff_gaussian_rasterization_ms import SparseGaussianAdam

TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             args):
    torch.set_deterministic_debug_mode(1)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(sh_degree=0)

    scene = Scene(dataset, gaussians, resolution_scales=[1, 2], shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    ext.my_logger.basic_config()
    db_cfg = NERF_DATASETS['Mip360']['common'] | NERF_DATASETS['Mip360']['train'] | {
        'coord_src': 'colmap',
        'coord_dst': 'colmap',
        'batch_mode': True,
        'with_rays': False,
        'background': 'black',
        'near': 0.01,
        'far': 100.0,
        'downscale': 4,
    }
    db_cfg['root'] = Path('~/data').expanduser().joinpath(db_cfg['root'])
    db = ColmapDataset(**db_cfg)
    model = MiniSplatting2(
        adaptive_control_cfg=dict(
            aggressive_clone_interval=[250, 500, -1],
            densify_interval=[100, 500, 3_000],
        ),
        warmup_step=3000,
        lr_opacity=25.,
        q_xyzw=False,
    )
    model.create_from_pcd(scene.pcd, scene.cameras_extent)
    model.set_from_dataset(db)
    model = model.cuda()
    optimizer = SparseGaussianAdam(model.get_params(utils.Config(lr=1e-3)), lr=1e-3, eps=1e-15)
    model._task = utils.Config(optimizer=optimizer, cfg=utils.Config(epochs=opt.iterations), train_db=db)
    model.training_setup()
    print(model)

    def copy_to_model(copy_buffer=False):
        for param_name, opt_name in model.param_names_map.items():
            v = getattr(gaussians, param_name)
            # if param_name == '_rotation':
            #     v = v[:, (1, 2, 3, 0)]  # wxyz -> xyzw
            v2 = getattr(model, param_name)
            if v.numel() > 0:
                v2.data.copy_(v)
        if copy_buffer:
            model._culling.data.copy_(gaussians._culling)
            model.factor_culling.data.copy_(gaussians.factor_culling)
            model.denom.data.copy_(gaussians.denom)
            model.max_radii2D.data.copy_(gaussians.max_radii2D)
            model.xyz_gradient_accum.data.copy_(gaussians.xyz_gradient_accum)
            model.mask_blur.copy_(mask_blur)

    def copy_optimizer_state():
        for param_name, opt_name in model.param_names_map.items():
            v = None
            for g in gaussians.optimizer.param_groups:
                if g['name'] == opt_name:
                    v = gaussians.optimizer.state.get(g['params'][0], None)
                    break
            v2 = None
            for g in optimizer.param_groups:
                if g['name'] == opt_name:
                    v2 = optimizer.state.get(g['params'][0], None)
                    break
            if v is not None and v2 is not None:
                v2["exp_avg"].data.copy_(v["exp_avg"])
                v2["exp_avg_sq"].data.copy_(v["exp_avg_sq"])

    def check_optimizer_state():
        for param_name, opt_name in model.param_names_map.items():
            v = None
            for g in gaussians.optimizer.param_groups:
                if g['name'] == opt_name:
                    v = gaussians.optimizer.state.get(g['params'][0], None)
                    break
            v2 = None
            for g in optimizer.param_groups:
                if g['name'] == opt_name:
                    v2 = optimizer.state.get(g['params'][0], None)
                    break
            if v is not None and v2 is not None:
                get_rel_error(v2["exp_avg"], v["exp_avg"], f"{param_name} exp_avg")
                get_rel_error(v2["exp_avg_sq"], v["exp_avg_sq"], f"{param_name} exp_avg_sq")

    copy_to_model(False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = scene.getTrainCameras_warn_up(0, args.warn_until_iter, scale=1.0, scale2=2.0).copy()

    for i, view in enumerate(viewpoint_stack):
        print(view.uid, view.image_name, db.image_names[i])
        assert view.uid == db.image_names.index(view.image_name + '.JPG'), \
            f"{view.uid} == {db.image_names.index(view.image_name + '.JPG')}"

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    mask_blur = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool, device='cuda')
    gaussians.init_culling(len(scene.getTrainCameras()))

    def check_model(threshold: float = None):
        for param_name in list(model.param_names_map.keys()) + [
            '_culling', 'factor_culling', 'max_radii2D', 'xyz_gradient_accum', 'denom', 'mask_blur'
        ]:
            v = getattr(gaussians, param_name) if param_name != 'mask_blur' else mask_blur
            # if param_name == '_rotation':
            #     v = v[:, (1, 2, 3, 0)]  # wxyz -> xyzw
            # print(param_name, utils.show_shape(v, getattr(model, param_name)))
            if v.numel() == 0:
                continue
            v2 = getattr(model, param_name)
            assert v.shape == v2.shape, f"[{iteration=}]: {param_name} {v.shape} vs {v2.shape}"
            assert v.dtype == v2.dtype, f"[{iteration=}]: {param_name} {v.dtype} vs {v2.dtype}"
            if v.dtype.is_floating_point:
                if v.dtype.is_floating_point:
                    error = (v - v2).abs().max()
                    if error > 1e-4:
                        print(f"[{iteration=}]: {param_name} {error.item():.4e}")
                    if threshold is not None and param_name != 'factor_culling':
                        assert error < threshold
                else:
                    error = (v ^ v2).sum()
                    if error > 0:
                        print(f"[{iteration=}]: {param_name} {error.item()}")
            # get_abs_error(v.float(), getattr(model, param_name).float(), param_name, threshold=threshold)

    def check_grad_lr(threshold: float = None):
        for param_name, opt_name in model.param_names_map.items():
            v = None
            for g in gaussians.optimizer.param_groups:
                if g['name'] == opt_name:
                    v = g
                    break
            v2 = None
            for g in optimizer.param_groups:
                if g['name'] == opt_name:
                    v2 = g
                    break
            if v is not None and v2 is not None and v['params'][0].numel() > 0:
                g1, g2 = v2['params'][0].grad, v['params'][0].grad
                # if param_name == '_rotation':
                #     g2 = g2[:, (1, 2, 3, 0)]
                error = (g1 - g2).abs().max()
                if error > 1e-4:
                    print(f"[{iteration=}]: {param_name} {error.item():.4e}")
                if threshold is not None:
                    assert error < threshold
                # get_rel_error(g1, g2, f"grad {param_name}", threshold=1.)
                assert v2['lr'] == v['lr'], f"{param_name}: {v2['lr']} vs {v['lr']}"

    for iteration in range(first_iter, opt.iterations + 1):
        print(f"{iteration=:=^40d}")
        iter_start.record()

        copy_to_model(True)
        copy_optimizer_state()
        gaussians.update_learning_rate(iteration)
        model.change_with_training_progress(iteration - 1, opt.iterations, 0, 1)
        model.update_learning_rate(iteration)

        if iteration % 1000 == 0 and iteration > args.simp_iteration1:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras_warn_up(
                iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render_imp(viewpoint_cam, gaussians, pipe, background,
                                culling=gaussians._culling[:, viewpoint_cam.uid])

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]

        infos = {
            'Tw2v': viewpoint_cam.world_view_transform.T,
            'Tv2c': viewpoint_cam.projection_matrix.T,
            'campos': viewpoint_cam.camera_center,
            'size': torch.tensor([viewpoint_cam.image_width, viewpoint_cam.image_height], device='cuda'),
            'index': torch.tensor(viewpoint_cam.uid, device='cuda'),
            'FoV': torch.tensor([viewpoint_cam.FoVx, viewpoint_cam.FoVy], device='cuda', dtype=torch.double),
        }
        # print(utils.show_shape(infos))
        outputs = model.render(background=background, info=infos, HWC=False)
        outputs['render'] = outputs.pop('images')  # .permute(0, 3, 2, 1)

        def check_outputs(out1, out2, threshold: float = None, show=False):
            for k, v in out1.items():
                if k in outputs:
                    v2 = out2[k]
                    # assert v.shape == v2.shape, f"{k}, {v.shape} {v2.shape}"
                    if isinstance(v, torch.Tensor):
                        if v.numel() > 0:
                            assert v.shape == v2.shape, f"[{iteration=}]: {k} {v.shape} vs {v2.shape}"
                            if v.dtype.is_floating_point:
                                error = (v - v2).abs().max() / max(1., v.abs().max().item())
                                if show or error > 1e-4:
                                    print(f"[{iteration=}]: {k} {error.item():.4e}")
                                if threshold is not None:
                                    assert error < threshold
                            else:
                                error = (v ^ v2).sum()
                                if show or error > 0:
                                    print(f"[{iteration=}]: {k} {error.item()}")
                            # get_abs_error(v2.float(), v.float(), k, threshold=threshold)
                #     else:
                #         print(k, utils.show_shape(v, v2))
                # else:
                #     print(k, utils.show_shape(v))

        # if iteration >= 500:
        #     print(utils.show_shape(outputs['visibility_filter'], visibility_filter))
        #     for k in set(list(outputs.keys()) +list(render_pkg.keys())):
        #         v1 = outputs.get(k, None)
        #         v2 = render_pkg.get(k, None)
        #         if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor) and v1.shape == v2.shape:
        #             get_rel_error(v1.float(), v2.float(), k)
        #         else:
        #             print(k, utils.show_shape(v1, v2))
        #     show_max_different(model._culling.float(), gaussians._culling.float())
        check_outputs(render_pkg, outputs, threshold=1e-2)

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        Ll1_ = l1_loss(outputs['render'], gt_image)
        ssim_ = fused_ssim(outputs['render'].unsqueeze(0), gt_image.unsqueeze(0))
        loss_ = (1.0 - opt.lambda_dssim) * Ll1_ + opt.lambda_dssim * (1.0 - ssim_)
        loss_.backward()
        print(f'loss diff: {loss_.item() - loss.item():.6e}')
        check_grad_lr(threshold=1e-2)

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
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            mask = radii > 0
            viewspace_point_tensor = outputs['viewspace_points']
            cfg = model.adaptive_control_cfg
            densify_interval = model.adaptive_control_cfg['densify_interval']
            prune_interval = model.adaptive_control_cfg['prune_interval']
            # max_step = max(densify_interval[2], prune_interval[2])
            max_step = densify_interval[2]
            step = iteration  # start from 1

            print('opt.densify_until_iter', max_step, opt.densify_until_iter)
            # # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                model.max_radii2D[mask] = torch.max(model.max_radii2D[mask], radii[mask])
                if gaussians._culling[:, viewpoint_cam.uid].sum() == 0:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                else:
                    # normalize xy gradient after culling
                    gaussians.add_densification_stats_culling(viewspace_point_tensor, visibility_filter,
                                                              gaussians.factor_culling)
                if outputs['num_cull'] == 0:
                    model.add_densification_stats(viewspace_point_tensor, mask)
                else:
                    # normalize xy gradient after culling
                    model.add_densification_stats_culling(viewspace_point_tensor, mask, model.factor_culling)

                area_max = render_pkg["area_max"]
                mask_blur = torch.logical_or(mask_blur, area_max > (image.shape[1] * image.shape[2] / 5000))

                area_max_ = outputs["area_max"]
                image_ = outputs['render']
                model.mask_blur = torch.logical_or(model.mask_blur,
                                                   area_max_ > (image_.shape[1] * image_.shape[2] / 5000))

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and iteration != args.depth_reinit_iter:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    masks = gaussians.densify_and_prune_mask(opt.densify_grad_threshold,
                                                             0.005, scene.cameras_extent,
                                                             size_threshold, mask_blur)
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool, device='cuda')

                    masks2 = model.densify_and_prune_mask(
                        optimizer, cfg['densify_grad_threshold'], 0.005, model.cameras_extent, size_threshold,
                        model.mask_blur, cfg['densify_percent_dense']
                    )
                    model.mask_blur = torch.zeros(model._xyz.shape[0], dtype=torch.bool, device='cuda')
                    # print(gaussians._xyz.shape)
                    print('densify_and_prune_mask')
                    for m1, m2 in zip(masks, masks2):
                        if m1.shape == m2.shape:
                            print((m1 ^ m2).sum() if m1.dtype == torch.bool else (m1 - m2).abs().max())
                        else:
                            print(utils.show_shape(m1, m2))

                    check_model(threshold=1e-2)
                if iteration == args.depth_reinit_iter:
                    num_depth = gaussians._xyz.shape[0] * args.num_depth_factor
                    print('num_depth', gaussians._xyz.shape[0], args.num_depth_factor, num_depth)

                    # interesction_preserving for better point cloud reconstruction result at the early stage, not affect rendering quality
                    p1, r1 = gaussians.intersection_preserving(scene, render_simp, iteration, args, pipe, background)
                    pts, rgb = gaussians.depth_reinit(scene, render_depth, iteration, num_depth, args, pipe, background)

                    gaussians.reinitial_pts(pts, rgb)

                    gaussians.training_setup(opt)
                    gaussians.init_culling(len(scene.getTrainCameras()))
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool, device='cuda')

                    views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()
                    p2, r2 = model.interesction_preserving(optimizer, step, views)
                    pts_, rgb_ = model.depth_reinit(step, num_depth, views)
                    model.reinitial_pts(optimizer, pts_, rgb_)
                    model.training_setup()
                    torch.cuda.empty_cache()
                    print('depth_reinit')
                    get_rel_error(p2, p1, 'interesction_preserving xyz')
                    get_rel_error(r2, r1, 'interesction_preserving rgb')
                    get_rel_error(pts_, pts, 'pts')
                    get_rel_error(rgb_, rgb, 'rgb')
                    check_model(threshold=1e-2)
                    # print(gaussians._xyz.shape)

                if iteration >= args.aggressive_clone_from_iter and iteration % args.aggressive_clone_interval == 0 and iteration != args.depth_reinit_iter:
                    for param_name, opt_name in model.param_names_map.items():
                        v = None
                        for g in gaussians.optimizer.param_groups:
                            if g['name'] == opt_name:
                                v = g
                                break
                        v2 = None
                        for g in optimizer.param_groups:
                            if g['name'] == opt_name:
                                v2 = g
                                break
                        if v is not None and v2 is not None and v['params'][0].numel() > 0:
                            v, v2 = v['params'][0], v2['params'][0]
                            # if param_name == '_rotation':
                            #     v = v[:, (1, 2, 3, 0)]
                            get_rel_error(v, v2, f"{param_name}", threshold=1.)

                    o2 = model.culling_with_clone(optimizer,
                                                  scene.getTrainCameras_warn_up(iteration, args.warn_until_iter,
                                                                                scale=1.0, scale2=2.0).copy())
                    torch.cuda.empty_cache()
                    model.mask_blur = model.mask_blur.new_zeros(model._xyz.shape[0])

                    o1 = gaussians.culling_with_clone(scene, render_simp, iteration, args, pipe, background)
                    torch.cuda.empty_cache()
                    mask_blur = mask_blur.new_zeros(gaussians._xyz.shape[0])
                    print('culling_with_clone')
                    # print(gaussians._xyz.shape)
                    # print(utils.show_shape(mask_blur, model.mask_blur))
                    for v1, v2, name in zip(o1, o2, ['count_vis', 'count_rad', 'imp_score', 'accum_area_max',
                                                     'prune_mask', 'intersection_pts_mask',
                                                     ]):
                        get_rel_error(v1.float(), v2.float(), name)
                    get_rel_error(o1[0] / (o1[1] + 1e-1), o2[0] / (o2[1] + 1e-1), 'factor')
                    check_model(threshold=1e-2)
                    # exit()

            model.hook_after_train_step()
            if iteration == args.simp_iteration1:
                gaussians.culling_with_intersection_sampling(scene, render_simp, iteration, args, pipe, background)
                gaussians.max_sh_degree = dataset.sh_degree
                gaussians.extend_features_rest()

                gaussians.training_setup(opt)
                torch.cuda.empty_cache()
                print('culling_with_interesction_sampling offical')
                # print(gaussians._xyz.shape)

            if iteration == args.simp_iteration2:
                gaussians.culling_with_intersection_preserving(scene, render_simp, iteration, args, pipe, background)
                torch.cuda.empty_cache()
                print('culling_with_interesction_preserving offical')
                # print(gaussians._xyz.shape)

            if iteration == (args.simp_iteration2 + opt.iterations) // 2:
                gaussians.init_culling(len(scene.getTrainCameras()))
                print('init_culling offical')

            # Optimizer step
            # if iteration > 600:
            #     print('...')
            #     for param_name, opt_name in model.param_names_map.items():
            #         v = None
            #         for g in gaussians.optimizer.param_groups:
            #             if g['name'] == opt_name:
            #                 v = g['params'][0]
            #                 break
            #         v2 = None
            #         for g in optimizer.param_groups:
            #             if g['name'] == opt_name:
            #                 v2 = g['params'][0]
            #                 break
            #         if v.numel() == 0:
            #             continue
            #         get_rel_error(v, v2, param_name)
            #         s1 = gaussians.optimizer.state.get(v, None)
            #         s2 = gaussians.optimizer.state.get(v2, None)
            #         print(utils.show_shape(s1, s2))
            #         if s1 is not None and s2 is not None:
            #             get_rel_error(s2["exp_avg"], s1["exp_avg"], f"{g['name']} exp_avg")
            #             get_rel_error(s2["exp_avg_sq"], s1["exp_avg_sq"], f"{g['name']} exp_avg_sq")
            if iteration < opt.iterations:
                visible = radii > 0
                gaussians.optimizer.step(visible, radii.shape[0])
                gaussians.optimizer.zero_grad(set_to_none=True)

                optimizer.step(visible, radii.shape[0])
                optimizer.zero_grad(set_to_none=True)
            check_model(threshold=1e-2)
            check_optimizer_state()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    print('Num of Guassians: %d' % (gaussians._xyz.shape[0]))
    return


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
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},)

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssims = []
                lpipss = []
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    ssims.append(ssim(image, gt_image))
                    lpipss.append(lpips(image, gt_image, net_type='vgg'))

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])

                ssims_test = torch.tensor(ssims).mean()
                lpipss_test = torch.tensor(lpipss).mean()

                print("\n[ITER {}] Evaluating {}: ".format(iteration, config['name']))
                print("  SSIM : {:>12.7f}".format(ssims_test.mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(psnr_test.mean(), ".5"))
                print("  LPIPS : {:>12.7f}".format(lpipss_test.mean(), ".5"))
                print("")

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    parser.add_argument("--imp_metric", required=True, type=str, default=None)

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
    if not -1 in args.test_iterations:
        args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    torch.cuda.synchronize()
    time_start = time.time()

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    torch.cuda.synchronize()
    time_end = time.time()
    time_total = time_end - time_start
    print('time: %fs' % (time_total))

    time_txt_path = os.path.join(args.model_path, r'time.txt')
    with open(time_txt_path, 'w') as f:
        f.write(str(time_total))

        # All done
    print("\nTraining complete.")
