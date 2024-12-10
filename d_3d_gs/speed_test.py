import os
from argparse import ArgumentParser
from os import makedirs

import imageio
import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from gaussian_renderer import render
from scene import Scene, DeformModel
from utils.general_utils import safe_state


@torch.no_grad()
def interpolate_time(
    model_path, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform,
    num_frames=1000,
):
    render_path = os.path.join(model_path, name)
    makedirs(render_path, exist_ok=True)

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]
    renderings = []
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.synchronize()
    start_time.record()
    for t in tqdm(range(0, num_frames, 1), desc="Rendering progress"):
        fid = torch.Tensor([t / (num_frames - 1)]).cuda()
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        # renderings.append(results["render"])
    end_time.record()
    end_time.synchronize()
    t = start_time.elapsed_time(end_time)
    fps = num_frames / (t / 1000.)
    print(f"Rendering {num_frames} images of view {idx} in {t:.2f} ms, fps={fps:.2f}")
    # renderings = np.stack([to8b(img.cpu().numpy()) for img in renderings], 0).transpose(0, 2, 3, 1)
    # imageio.mimwrite(os.path.join(render_path, f'video_{iteration}.mp4'), renderings, fps=60, quality=8)  # noqa
    with open(os.path.join(model_path, name, "results.txt"), 'w') as f:
        f.write(f"FPS: {fps:.2f}\n")
        f.write(f"Time(ms): {t:.2f}\n")
        f.write(f"Frames: {num_frames}\n")
        f.write(f"View: {idx}\n")


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
        deform.load_weights(dataset.model_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        interpolate_time(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "speed", scene.loaded_iter,
                         scene.getTestCameras(), gaussians, pipeline, background, deform)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args))
