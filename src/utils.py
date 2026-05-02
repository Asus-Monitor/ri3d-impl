"""Shared utilities used across multiple pipeline steps.

Consolidates code that was duplicated across step files:
  - save_depth_vis: depth map visualization (was in step1, step2, step3)
  - prepare_for_pipeline: image resize for diffusion (was in step5)
  - estimate_mono_depth / clear_mono_depth_cache: monocular depth (was in step8)
  - load_gt_images: GT image loading at render resolution (was in step5, step6, step8)
  - load_mono_depths: mono depth loading with convention fix (was in step6, step8)
  - compute_scene_scale: from camera positions (was in step5, step6, step8)
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RI3DConfig


def save_depth_vis(depth: np.ndarray, path: Path, title: str = ""):
    """Save a depth map as a colorized PNG.

    Handles zero/invalid depths by computing percentiles only on valid pixels.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    valid = depth[depth > 0]
    if len(valid) > 0:
        vmin, vmax = np.percentile(valid, [2, 98])
    else:
        vmin, vmax = depth.min(), depth.max()
    im = ax.imshow(depth, cmap="turbo", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def prepare_for_pipeline(image: Image.Image, target_short_side: int = 512) -> tuple[Image.Image, int, int]:
    """Resize preserving aspect ratio for diffusion pipeline input.

    Per paper (Sec 4.2): "resize the input images during fine-tuning so that
    their smallest dimension (typically height) is 512 pixels".
    Dims are rounded to multiples of 8 for VAE latent alignment.

    Returns (resized_pil, pipe_h, pipe_w).
    """
    W_orig, H_orig = image.size
    if H_orig <= W_orig:
        pipe_h = target_short_side
        pipe_w = int(W_orig * target_short_side / H_orig)
    else:
        pipe_w = target_short_side
        pipe_h = int(H_orig * target_short_side / W_orig)
    pipe_h = (pipe_h // 8) * 8
    pipe_w = (pipe_w // 8) * 8
    resized = image.resize((pipe_w, pipe_h), Image.LANCZOS)
    return resized, pipe_h, pipe_w


def estimate_mono_depth(image_tensor: torch.Tensor, cfg: RI3DConfig) -> torch.Tensor:
    """Run monocular depth estimation on an image tensor.

    Returns proper depth (larger = farther). Depth Anything V2 outputs inverse
    depth (disparity), so we invert it here to match rendered depth convention.

    Uses a cached pipeline to avoid reloading the model on every call.

    Args:
        image_tensor: (H, W, 3) float tensor in [0, 1]
    Returns:
        depth: (H, W) float tensor, proper depth (larger = farther)
    """
    from transformers import pipeline as hf_pipeline

    H, W = image_tensor.shape[:2]
    img_np = (image_tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)

    if not hasattr(estimate_mono_depth, "_pipe"):
        estimate_mono_depth._pipe = hf_pipeline(
            "depth-estimation", model=cfg.depth_model,
            device=cfg.device, torch_dtype=cfg.dtype,
        )
    with torch.no_grad():
        result = estimate_mono_depth._pipe(img_pil)

    mono = result["predicted_depth"]
    if mono.dim() == 3:
        mono = mono.squeeze(0)
    mono = F.interpolate(
        mono.unsqueeze(0).unsqueeze(0).float(),
        size=(H, W), mode="bilinear", align_corners=False,
    ).squeeze()

    # Depth Anything V2 outputs inverse depth (closer = larger).
    # Convert to proper depth (closer = smaller).
    mono = 1.0 / (mono + 1e-6)

    return mono.to(image_tensor.device)


def clear_mono_depth_cache():
    """Free the cached Depth Anything pipeline from GPU."""
    if hasattr(estimate_mono_depth, "_pipe"):
        del estimate_mono_depth._pipe
        torch.cuda.empty_cache()


def load_gt_images(image_paths: list[str], H: int, W: int,
                   device: str = "cuda") -> list[torch.Tensor]:
    """Load GT images resized to (H, W) as float tensors on device."""
    gt_images = []
    for ip in image_paths:
        img = Image.open(ip).convert("RGB").resize((W, H), Image.LANCZOS)
        gt_images.append(torch.from_numpy(np.array(img)).float().to(device) / 255.0)
    return gt_images


def load_mono_depths(out_dir: Path, n_images: int,
                     device: str = "cuda") -> list[torch.Tensor]:
    """Load mono depths with convention correction (inverse depth -> proper depth).

    Uses DUSt3R depth as reference to detect and fix the convention.
    """
    from gaussian_trainer import ensure_depth_convention

    mono_depths = []
    for i in range(n_images):
        md = torch.load(out_dir / "mono_depths" / f"mono_depth_{i:03d}.pt", weights_only=True)
        dust3r_d = torch.load(out_dir / "dust3r_depths" / f"depth_{i:03d}.pt", weights_only=True)
        md = ensure_depth_convention(md.float(), dust3r_d.float())
        mono_depths.append(md.to(device))
    return mono_depths


def compute_scene_scale(poses: torch.Tensor) -> float:
    """Compute scene scale from camera positions (for densification thresholds)."""
    cam_positions = poses[:, :3, 3]
    scale = (cam_positions - cam_positions.mean(dim=0)).norm(dim=1).mean().item()
    return max(scale, 0.1)


def load_scene_data(cfg: RI3DConfig):
    """Load the per-scene tensors shared by stage1/stage2 optimization.

    Returns dict with: image_paths, poses, intrinsics, n_images, H, W,
    gt_images, mono_depths, out_dir.
    """
    out_dir = cfg.scene_output_dir()
    device = cfg.device

    image_paths = cfg.load_image_paths()
    poses = torch.load(out_dir / "dust3r_poses.pt", weights_only=True).float().to(device)
    intrinsics = torch.load(out_dir / "dust3r_intrinsics.pt", weights_only=True).float().to(device)
    n_images = len(image_paths)

    fused_depth_0 = torch.load(out_dir / "fused_depths" / "fused_depth_000.pt", weights_only=True)
    H, W = fused_depth_0.shape

    gt_images = load_gt_images(image_paths, H, W, device)
    mono_depths = load_mono_depths(out_dir, n_images, device)

    return {
        "out_dir": out_dir,
        "image_paths": image_paths,
        "poses": poses,
        "intrinsics": intrinsics,
        "n_images": n_images,
        "H": H,
        "W": W,
        "gt_images": gt_images,
        "mono_depths": mono_depths,
    }
