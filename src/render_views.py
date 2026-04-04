"""Render novel views from a trained checkpoint.

Usage:
  python render_views.py --checkpoint outputs/garden/checkpoints/final_checkpoint.pt --n_views 36
"""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RI3DConfig
from gaussian_trainer import GaussianModel
from step4_gaussian_init import generate_elliptical_cameras


def render_orbit(checkpoint_path: str, n_views: int = 36, output_dir: str = None):
    device = "cuda"
    ckpt = torch.load(checkpoint_path, weights_only=True)

    poses = ckpt["poses"].float().to(device)
    intrinsics = ckpt["intrinsics"].float().to(device)

    # Infer resolution from Gaussian data
    K_avg = intrinsics.mean(dim=0)
    H = int(K_avg[1, 2].item() * 2)
    W = int(K_avg[0, 2].item() * 2)
    H = max(H, 256)
    W = max(W, 256)

    if output_dir is None:
        output_dir = str(Path(checkpoint_path).parent.parent / "orbit_renders")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = GaussianModel(ckpt["gaussians"], device)
    from step4_gaussian_init import compute_scene_center
    scene_center = compute_scene_center(poses, ckpt["gaussians"]["means"].to(device))
    orbit_c2w = generate_elliptical_cameras(poses, n_views, scene_center).to(device)

    print(f"Rendering {n_views} orbit views at {H}x{W}...")

    for i in range(n_views):
        w2c = torch.linalg.inv(orbit_c2w[i])
        with torch.no_grad():
            r = model.render(w2c, K_avg, H, W)
        img = r["image"].clamp(0, 1).cpu().numpy()
        plt.imsave(out / f"orbit_{i:04d}.png", img)

    print(f"Saved {n_views} renders to {out}")
    print("To create video: ffmpeg -framerate 15 -i orbit_%04d.png -c:v libx264 -pix_fmt yuv420p orbit.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render orbit views from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_views", type=int, default=36)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    render_orbit(args.checkpoint, args.n_views, args.output)
