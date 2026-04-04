"""Step 2: Monocular depth estimation using Depth Anything V2 Small.

Outputs per scene:
  - outputs/<scene>/mono_depths/  per-view relative depth maps (.pt and .png)
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RI3DConfig


def save_depth_vis(depth: np.ndarray, path: Path, title: str = ""):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    vmin, vmax = np.percentile(depth, [2, 98])
    im = ax.imshow(depth, cmap="turbo", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_mono_depth(cfg: RI3DConfig, depth_pipe=None):
    from transformers import pipeline as hf_pipeline

    out_dir = cfg.scene_output_dir()
    mono_dir = out_dir / "mono_depths"
    mono_dir.mkdir(parents=True, exist_ok=True)

    # Load image paths from step 1
    image_paths = torch.load(out_dir / "image_paths.pt", weights_only=False)
    n_images = len(image_paths)

    # Load depth estimation pipeline if not provided externally
    _owns_pipe = depth_pipe is None
    if _owns_pipe:
        print(f"Loading Depth Anything V2 Small...")
        depth_pipe = hf_pipeline(
            "depth-estimation",
            model=cfg.depth_model,
            device=cfg.device,
            torch_dtype=cfg.dtype,
        )
        print("Model loaded.")

    # Also load the DUSt3R depth resolution for reference
    dust3r_depth_0 = torch.load(out_dir / "dust3r_depths" / "depth_000.pt", weights_only=True)
    target_h, target_w = dust3r_depth_0.shape
    print(f"Target resolution (matching DUSt3R): {target_h} x {target_w}")

    for i, img_path in enumerate(image_paths):
        name = Path(img_path).stem
        print(f"  Processing view {i} ({name})...")

        img = Image.open(img_path).convert("RGB")

        # Run depth estimation
        result = depth_pipe(img)
        # result["depth"] is a PIL Image, result["predicted_depth"] is tensor
        depth_tensor = result["predicted_depth"]  # (1, H, W) or (H, W)
        if depth_tensor.dim() == 3:
            depth_tensor = depth_tensor.squeeze(0)

        # Resize to match DUSt3R depth resolution using bilinear interpolation
        depth_resized = torch.nn.functional.interpolate(
            depth_tensor.unsqueeze(0).unsqueeze(0).float(),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        # Save tensor
        torch.save(depth_resized.cpu(), mono_dir / f"mono_depth_{i:03d}.pt")

        # Save visualization
        depth_np = depth_resized.cpu().numpy()
        save_depth_vis(depth_np, mono_dir / f"mono_depth_{i:03d}_{name}.png",
                       f"Mono Depth - View {i}")

        print(f"    Depth range: [{depth_np.min():.3f}, {depth_np.max():.3f}]")

    # Cleanup only if we loaded it ourselves
    if _owns_pipe:
        del depth_pipe
    torch.cuda.empty_cache()

    print(f"\nStep 2 complete! Mono depth maps in {mono_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Monocular depth estimation")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene image directory")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    cfg = RI3DConfig(scene_dir=Path(args.scene), output_dir=Path(args.output))
    run_mono_depth(cfg)
