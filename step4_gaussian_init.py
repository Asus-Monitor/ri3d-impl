"""Step 4: Initialize 3D Gaussians from fused depth maps and render initial views.

Per the paper (Sec 4.1):
  - Assign one Gaussian per pixel of every input image
  - Project into 3D along the ray using fused depth
  - Initialize color from pixel, rotation=identity, isometric scale=1.4*pixel_size, opacity=0.1

Outputs:
  - outputs/<scene>/init_gaussians.pt  (dict of Gaussian parameters)
  - outputs/<scene>/init_renders/      initial renders from input + novel views
"""
import argparse
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RI3DConfig


def unproject_depth(depth: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """Unproject depth map to 3D world coordinates.

    Args:
        depth: (H, W) depth map
        K: (3, 3) intrinsics
        c2w: (4, 4) camera-to-world transform

    Returns:
        points: (H*W, 3) world coordinates
    """
    H, W = depth.shape
    device = depth.device

    # Pixel grid
    v, u = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                           torch.arange(W, device=device, dtype=torch.float32),
                           indexing="ij")

    # Unproject to camera coords
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_cam = (u - cx) / fx * depth
    y_cam = (v - cy) / fy * depth
    z_cam = depth

    # Stack: (H, W, 3) -> (N, 3)
    pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1).reshape(-1, 3)

    # Transform to world: p_world = R @ p_cam + t
    R = c2w[:3, :3]  # (3, 3)
    t = c2w[:3, 3]   # (3,)
    pts_world = (pts_cam @ R.T) + t

    return pts_world


def compute_pixel_size(depth: float, fx: float) -> float:
    """Approximate size of a pixel in world units at given depth."""
    return depth / fx


def generate_elliptical_cameras(poses: torch.Tensor, n_cameras: int) -> torch.Tensor:
    """Generate novel cameras along an elliptical path aligned with input cameras.

    Args:
        poses: (N, 4, 4) input camera-to-world matrices
        n_cameras: number of novel cameras to generate

    Returns:
        novel_c2w: (n_cameras, 4, 4) camera-to-world matrices
    """
    # Compute center and mean "up" direction from input cameras
    positions = poses[:, :3, 3]  # (N, 3)
    center = positions.mean(dim=0)  # (3,)

    # Mean forward direction (cameras look along -Z in their local frame)
    forwards = -poses[:, :3, 2]  # (N, 3)
    mean_forward = forwards.mean(dim=0)
    mean_forward = mean_forward / mean_forward.norm()

    # Mean up direction
    ups = poses[:, :3, 1]  # (N, 3)
    mean_up = ups.mean(dim=0)
    mean_up = mean_up / mean_up.norm()

    # Right vector
    right = torch.cross(mean_forward, mean_up)
    right = right / right.norm()

    # Recompute up to ensure orthogonality
    mean_up = torch.cross(right, mean_forward)
    mean_up = mean_up / mean_up.norm()

    # Compute ellipse radii from spread of input cameras
    offsets = positions - center
    r_right = (offsets @ right).abs().max().item() * 1.2
    r_forward = (offsets @ mean_forward).abs().max().item() * 1.2
    r = max(r_right, r_forward, 0.5)

    # Mean distance from center
    mean_height = (offsets @ mean_up).mean().item()

    # Generate cameras on ellipse
    angles = torch.linspace(0, 2 * math.pi, n_cameras + 1, device=poses.device)[:-1]
    novel_c2w = torch.zeros(n_cameras, 4, 4, device=poses.device)

    for i, angle in enumerate(angles):
        # Position on ellipse
        pos = center + r * torch.cos(angle) * right + r * torch.sin(angle) * mean_forward
        pos = pos + mean_height * mean_up

        # Look at center
        forward = center - pos
        forward = forward / forward.norm()
        r_vec = torch.cross(forward, mean_up)
        r_vec = r_vec / r_vec.norm()
        up = torch.cross(r_vec, forward)
        up = up / up.norm()

        # Camera-to-world: columns are right, up, -forward, position
        # Convention: X=right, Y=up, Z=-forward (OpenGL)
        novel_c2w[i, :3, 0] = r_vec
        novel_c2w[i, :3, 1] = up
        novel_c2w[i, :3, 2] = -forward
        novel_c2w[i, :3, 3] = pos
        novel_c2w[i, 3, 3] = 1.0

    return novel_c2w


def init_gaussians(cfg: RI3DConfig):
    """Initialize Gaussians from fused depth maps."""
    out_dir = cfg.scene_output_dir()

    image_paths = torch.load(out_dir / "image_paths.pt", weights_only=False)
    poses = torch.load(out_dir / "dust3r_poses.pt", weights_only=True)      # (N, 4, 4) c2w
    intrinsics = torch.load(out_dir / "dust3r_intrinsics.pt", weights_only=True)  # (N, 3, 3)
    n_images = len(image_paths)

    all_means = []
    all_colors = []
    all_scales = []

    for i in range(n_images):
        name = Path(image_paths[i]).stem
        print(f"  Unprojecting view {i} ({name})...")

        # Load fused depth
        fused_depth = torch.load(out_dir / "fused_depths" / f"fused_depth_{i:03d}.pt",
                                  weights_only=True).float()
        H, W = fused_depth.shape
        K = intrinsics[i].float()
        c2w = poses[i].float()
        fx = K[0, 0].item()

        # Load image and resize to depth resolution
        img = Image.open(image_paths[i]).convert("RGB")
        img_np = np.array(img.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)  # (H, W, 3)

        # Unproject all pixels
        pts = unproject_depth(fused_depth, K, c2w)  # (H*W, 3)
        colors = img_tensor.reshape(-1, 3)            # (H*W, 3)

        # Filter out invalid depths (negative or very small)
        valid = fused_depth.reshape(-1) > 0.01
        pts = pts[valid]
        colors = colors[valid]
        depths_valid = fused_depth.reshape(-1)[valid]

        # Compute isometric scale: 1.4 * pixel_size
        pixel_sizes = depths_valid / fx * cfg.gaussian_scale_factor
        scales = pixel_sizes.unsqueeze(-1).expand(-1, 3)  # (N, 3)

        all_means.append(pts)
        all_colors.append(colors)
        all_scales.append(scales)

        print(f"    {pts.shape[0]} valid Gaussians, depth range "
              f"[{depths_valid.min():.2f}, {depths_valid.max():.2f}]")

    # Concatenate all
    means = torch.cat(all_means, dim=0)
    colors = torch.cat(all_colors, dim=0)
    scales = torch.cat(all_scales, dim=0)
    n_total = means.shape[0]
    print(f"\nTotal Gaussians: {n_total}")

    # Initialize other parameters
    quats = torch.zeros(n_total, 4)
    quats[:, 0] = 1.0  # identity rotation (wxyz)
    opacities = torch.full((n_total,), cfg.gaussian_init_opacity)

    # Store in log/logit space for optimization
    gaussians = {
        "means": means,
        "scales": torch.log(scales.clamp(min=1e-8)),  # log-space
        "quats": quats,
        "opacities": torch.logit(opacities.clamp(1e-4, 1 - 1e-4)),  # logit-space
        "colors": colors,
    }

    # Save
    torch.save(gaussians, out_dir / "init_gaussians.pt")
    print(f"Saved initial Gaussians to {out_dir / 'init_gaussians.pt'}")

    return gaussians


def render_initial_views(cfg: RI3DConfig):
    """Render initial views to verify Gaussian initialization."""
    from gsplat import rasterization

    out_dir = cfg.scene_output_dir()
    render_dir = out_dir / "init_renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device

    # Load data
    gaussians = torch.load(out_dir / "init_gaussians.pt", weights_only=True)
    poses = torch.load(out_dir / "dust3r_poses.pt", weights_only=True).float().to(device)
    intrinsics = torch.load(out_dir / "dust3r_intrinsics.pt", weights_only=True).float().to(device)
    image_paths = torch.load(out_dir / "image_paths.pt", weights_only=False)

    # Move to device
    means = gaussians["means"].float().to(device)
    scales = torch.exp(gaussians["scales"].float().to(device))
    quats = gaussians["quats"].float().to(device)
    opacities = torch.sigmoid(gaussians["opacities"].float().to(device))
    colors = gaussians["colors"].float().to(device)

    # Get render resolution from depth map
    fused_depth = torch.load(out_dir / "fused_depths" / "fused_depth_000.pt", weights_only=True)
    H, W = fused_depth.shape

    print(f"Rendering {len(image_paths)} input views at {H}x{W}...")

    # Render input views
    for i in range(len(image_paths)):
        name = Path(image_paths[i]).stem
        w2c = torch.linalg.inv(poses[i]).unsqueeze(0)  # (1, 4, 4)
        K = intrinsics[i].unsqueeze(0)                   # (1, 3, 3)

        with torch.no_grad():
            render_colors, render_alphas, _ = rasterization(
                means=means, quats=quats, scales=scales,
                opacities=opacities, colors=colors,
                viewmats=w2c, Ks=K, width=W, height=H,
                sh_degree=None, packed=True,
                near_plane=0.01, far_plane=1000.0,
            )

        img = render_colors[0].clamp(0, 1).cpu().numpy()
        alpha = render_alphas[0, :, :, 0].cpu().numpy()

        # Save render
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Render - View {i}")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(alpha, cmap="gray")
        plt.title(f"Alpha - View {i}")
        plt.axis("off")
        plt.savefig(render_dir / f"init_render_{i:03d}_{name}.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  View {i} ({name}): alpha coverage = {(alpha > 0.5).mean()*100:.1f}%")

    # Also render a few novel views
    print("\nRendering novel views...")
    novel_c2w = generate_elliptical_cameras(poses, n_cameras=4)
    K_avg = intrinsics.mean(dim=0)  # use average intrinsics for novel views

    for j in range(4):
        w2c = torch.linalg.inv(novel_c2w[j]).unsqueeze(0).to(device)
        K = K_avg.unsqueeze(0)

        with torch.no_grad():
            render_colors, render_alphas, _ = rasterization(
                means=means, quats=quats, scales=scales,
                opacities=opacities, colors=colors,
                viewmats=w2c, Ks=K, width=W, height=H,
                sh_degree=None, packed=True,
                near_plane=0.01, far_plane=1000.0,
            )

        img = render_colors[0].clamp(0, 1).cpu().numpy()
        alpha = render_alphas[0, :, :, 0].cpu().numpy()

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Novel View {j}")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(alpha, cmap="gray")
        plt.title(f"Alpha - Novel {j}")
        plt.axis("off")
        plt.savefig(render_dir / f"init_novel_{j:03d}.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  Novel view {j}: alpha coverage = {(alpha > 0.5).mean()*100:.1f}%")

    # Save novel cameras for later use
    torch.save(novel_c2w.cpu(), out_dir / "novel_cameras.pt")

    torch.cuda.empty_cache()
    print(f"\nStep 4 complete! Initial renders in {render_dir}")


def run_step4(cfg: RI3DConfig):
    init_gaussians(cfg)
    render_initial_views(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: Gaussian initialization + render")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene image directory")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    cfg = RI3DConfig(scene_dir=Path(args.scene), output_dir=Path(args.output))
    run_step4(cfg)
