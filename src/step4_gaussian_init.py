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


def compute_scene_center(poses: torch.Tensor, points: torch.Tensor = None) -> torch.Tensor:
    """Compute scene center as the average look-at point of input cameras.

    For each camera, finds the median-depth point along its viewing ray within
    the point cloud.  This is robust to distant background Gaussians that would
    pull a simple point-cloud centroid away from the actual subject.

    Args:
        poses: (N, 4, 4) camera-to-world matrices (OpenCV: +Z forward)
        points: (P, 3) 3D points (e.g. Gaussian means). If None, the scene
                center is estimated purely from camera geometry.

    Returns:
        scene_center: (3,) world-space center
    """
    positions = poses[:, :3, 3]
    cam_center = positions.mean(dim=0)

    if points is not None and len(points) > 0:
        look_ats = []
        for i in range(len(poses)):
            pos = positions[i]
            fwd = poses[i, :3, 2]
            fwd = fwd / fwd.norm()
            # Project all points onto this camera's forward ray
            vecs = points - pos.unsqueeze(0)
            depths = (vecs * fwd.unsqueeze(0)).sum(dim=1)
            in_front = depths > 0.01
            if in_front.sum() > 0:
                median_depth = depths[in_front].median()
                look_ats.append(pos + median_depth * fwd)
        if look_ats:
            return torch.stack(look_ats).mean(dim=0)

    # Fallback: step forward from camera center along mean gaze
    forwards = poses[:, :3, 2]
    mean_forward = forwards.mean(dim=0)
    mean_forward = mean_forward / mean_forward.norm()
    cam_spread = (positions - cam_center).norm(dim=1).mean().item()
    return cam_center + max(cam_spread * 2.0, 0.5) * mean_forward


def generate_elliptical_cameras(poses: torch.Tensor, n_cameras: int,
                                 scene_center: torch.Tensor = None,
                                 restrict_to_inputs: bool = False,
                                 margin_deg: float = 20.0) -> torch.Tensor:
    """Generate novel cameras along an elliptical path aligned with input cameras.

    Per paper (Sec 4.3): "we introduce a set of M novel cameras along an elliptical
    path aligned with the input cameras."

    Strategy: orbit in the plane perpendicular to the camera "up" direction
    (a horizontal ring around the scene center). This is robust for any camera
    arrangement because cameras always have a consistent up direction from
    DUSt3R's global alignment.

    Uses OpenCV camera convention (DUSt3R output): +X right, +Y down, +Z forward.

    Args:
        poses: (N, 4, 4) input camera-to-world matrices (OpenCV convention)
        n_cameras: number of novel cameras to generate
        scene_center: (3,) optional scene center; if None, estimated from cameras
        restrict_to_inputs: if True, cameras are placed only within the angular
            range covered by input cameras (+ margin). Use True for Stage 1
            (visible-region reconstruction). Use False for Stage 2 (full coverage
            needed for inpainting missing regions).
        margin_deg: angular margin in degrees beyond the input range (each side).

    Returns:
        novel_c2w: (n_cameras, 4, 4) camera-to-world matrices
    """
    positions = poses[:, :3, 3]  # (N, 3)
    cam_center = positions.mean(dim=0)  # (3,)

    # DUSt3R OpenCV convention: cameras look along +Z in local frame
    forwards = poses[:, :3, 2]  # (N, 3)
    mean_forward = forwards.mean(dim=0)
    mean_forward = mean_forward / mean_forward.norm()

    # Estimate scene center if not provided
    if scene_center is None:
        cam_spread = (positions - cam_center).norm(dim=1).mean().item()
        depth_est = max(cam_spread * 2.0, 0.5)
        scene_center = cam_center + depth_est * mean_forward

    # Camera "up" direction (OpenCV: Y is down, so up = -Y column of c2w)
    mean_up = -poses[:, :3, 1].mean(dim=0)
    mean_up = mean_up / mean_up.norm()

    # --- Build orbit plane perpendicular to "up" (horizontal ring) ---
    # Direction from cameras toward scene, projected onto the horizontal plane
    look_dir = scene_center - cam_center
    look_horiz = look_dir - (look_dir @ mean_up) * mean_up
    if look_horiz.norm() < 1e-6:
        # Scene directly above/below cameras — use mean_forward as fallback
        look_horiz = mean_forward - (mean_forward @ mean_up) * mean_up
    look_horiz = look_horiz / look_horiz.norm()

    # Two orthogonal horizontal axes for the orbit
    axis1 = torch.linalg.cross(mean_up, look_horiz)  # perpendicular to look direction
    axis1 = axis1 / axis1.norm()
    axis2 = look_horiz  # toward the scene

    # Elliptical orbit radii: per-axis distances from scene center to input cameras.
    # Paper: "along an elliptical path aligned with the input cameras."
    # Computing separate radii along axis1 and axis2 produces an ellipse that
    # matches the input camera distribution (e.g., elongated for forward-facing scenes).
    sc_offsets = positions - scene_center.unsqueeze(0)  # (N, 3)
    horiz_offsets = sc_offsets - (sc_offsets @ mean_up).unsqueeze(-1) * mean_up.unsqueeze(0)
    proj1 = (horiz_offsets @ axis1).abs()  # distances along axis1
    proj2 = (horiz_offsets @ axis2).abs()  # distances along axis2
    radius1 = max(proj1.mean().item() * 1.1, 0.3)
    radius2 = max(proj2.mean().item() * 1.1, 0.3)

    # Height: median camera displacement from scene center along up direction
    # (median is more robust than mean when one camera is an outlier)
    mean_height = (sc_offsets @ mean_up).median().item()

    # --- Compute angular range ---
    if restrict_to_inputs:
        # Find the smallest arc on the orbit circle that contains all input cameras.
        # Novel cameras are distributed within this arc + margin, so they only
        # see regions actually covered by input views (where repair makes sense).
        input_angles = []
        for i in range(len(poses)):
            off = positions[i] - scene_center
            x = (off @ axis1).item()
            y = (off @ axis2).item()
            input_angles.append(math.atan2(y, x))

        sorted_angles = sorted(input_angles)
        n_inp = len(sorted_angles)

        # Find the largest gap between consecutive cameras (the "empty" region)
        max_gap = -1.0
        max_gap_idx = 0
        for k in range(n_inp):
            a_cur = sorted_angles[k]
            a_next = sorted_angles[(k + 1) % n_inp]
            gap = a_next - a_cur if k < n_inp - 1 else (a_next + 2 * math.pi - a_cur)
            if gap > max_gap:
                max_gap = gap
                max_gap_idx = k

        # The covered arc starts AFTER the largest gap
        arc_start = sorted_angles[(max_gap_idx + 1) % n_inp]
        arc_end = sorted_angles[max_gap_idx]
        if arc_end < arc_start:
            arc_end += 2 * math.pi

        margin_rad = math.radians(margin_deg)
        arc_start -= margin_rad
        arc_end += margin_rad

        # If the input cameras already span nearly 360°, just do full orbit
        arc_span = arc_end - arc_start
        if arc_span >= 2 * math.pi * 0.9:
            angles = torch.linspace(0, 2 * math.pi, n_cameras + 1, device=poses.device)[:-1]
        else:
            angles = torch.linspace(float(arc_start), float(arc_end),
                                    n_cameras + 2, device=poses.device)[1:-1]
    else:
        angles = torch.linspace(0, 2 * math.pi, n_cameras + 1, device=poses.device)[:-1]

    novel_c2w = torch.zeros(n_cameras, 4, 4, device=poses.device)

    for i, angle in enumerate(angles):
        # Position on elliptical orbit (per-axis radii)
        pos = (scene_center
               + radius1 * torch.cos(angle) * axis1
               + radius2 * torch.sin(angle) * axis2
               + mean_height * mean_up)

        # Look-at: forward direction from camera toward scene center
        fwd = scene_center - pos
        fwd = fwd / fwd.norm()

        # Build camera axes (OpenCV: X=right, Y=down, Z=forward)
        r_vec = torch.linalg.cross(fwd, mean_up)
        if r_vec.norm() < 1e-6:
            r_vec = axis1
        r_vec = r_vec / r_vec.norm()
        down = torch.linalg.cross(fwd, r_vec)
        # Ensure down points in the -up direction
        if down @ (-mean_up) < 0:
            down = -down
            r_vec = -r_vec

        # c2w columns: X=right, Y=down, Z=forward
        novel_c2w[i, :3, 0] = r_vec
        novel_c2w[i, :3, 1] = down
        novel_c2w[i, :3, 2] = fwd
        novel_c2w[i, :3, 3] = pos
        novel_c2w[i, 3, 3] = 1.0

    return novel_c2w


def init_gaussians(cfg: RI3DConfig):
    """Initialize Gaussians from fused depth maps."""
    out_dir = cfg.scene_output_dir()

    image_paths = cfg.load_image_paths()
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

        # Filter out invalid depths (negative, very small, NaN, inf)
        depth_flat = fused_depth.reshape(-1)
        valid = (depth_flat > 0.01) & torch.isfinite(depth_flat)
        pts = pts[valid]
        colors = colors[valid]
        depths_valid = depth_flat[valid]

        if pts.shape[0] == 0:
            print(f"    WARNING: No valid depths for view {i}, skipping")
            continue

        # Compute isometric scale: 1.4 * pixel_size
        pixel_sizes = depths_valid / fx * cfg.gaussian_scale_factor
        scales = pixel_sizes.unsqueeze(-1).expand(-1, 3)  # (N, 3)

        all_means.append(pts)
        all_colors.append(colors)
        all_scales.append(scales)

        print(f"    {pts.shape[0]} valid Gaussians, depth range "
              f"[{depths_valid.min():.2f}, {depths_valid.max():.2f}]")

    # Concatenate all
    if len(all_means) == 0:
        raise RuntimeError("No valid Gaussians from any view. Check depth fusion output.")
    means = torch.cat(all_means, dim=0)
    colors = torch.cat(all_colors, dim=0)
    scales = torch.cat(all_scales, dim=0)
    n_total = means.shape[0]
    print(f"\nTotal Gaussians: {n_total}")

    # Initialize other parameters
    quats = torch.zeros(n_total, 4)
    quats[:, 0] = 1.0  # identity rotation (wxyz)
    opacities = torch.full((n_total,), cfg.gaussian_init_opacity)

    # Convert RGB to SH DC coefficients. Paper §3.1: "color c represented with
    # spherical harmonics (SH) coefficients". Without SH, a Gaussian has one
    # fixed RGB regardless of viewing direction — multiple Gaussians from
    # different input views overlap at novel angles and blend incoherently,
    # producing "colored splat blobs" that are OOD for the repair model (real
    # 3DGS never renders like that). With SH degree 3, each Gaussian has 16
    # coefficients per channel; DC (index 0) encodes the base color and higher
    # orders encode view-dependent residuals learned during optimization.
    #
    # Gsplat renders: rendered_color = C0 * DC + 0.5 + higher_terms (see
    # gsplat.exporter.sh2rgb). Init DC so view-independent render == RGB:
    #   DC = (RGB - 0.5) / C0. We derive C0 from sh2rgb to stay in sync with
    # gsplat's convention.
    from gsplat.exporter import sh2rgb
    SH_C0 = float(sh2rgb(torch.tensor(1.0)) - sh2rgb(torch.tensor(0.0)))
    SH_DEGREE = 3
    K_coefs = (SH_DEGREE + 1) ** 2  # = 16 for degree 3
    sh_coefs = torch.zeros(n_total, K_coefs, 3)
    sh_coefs[:, 0, :] = (colors - 0.5) / SH_C0

    # Store in log/logit space for optimization
    gaussians = {
        "means": means,
        "scales": torch.log(scales.clamp(min=1e-8)),  # log-space
        "quats": quats,
        "opacities": torch.logit(opacities.clamp(1e-4, 1 - 1e-4)),  # logit-space
        "colors": sh_coefs,  # (N, 16, 3) SH coefficients, not raw RGB
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
    image_paths = cfg.load_image_paths()

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
                sh_degree=3, packed=True,
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
    # Compute scene center from camera look-at points (robust to distant Gaussians)
    scene_center = compute_scene_center(poses, means)
    novel_c2w = generate_elliptical_cameras(poses, n_cameras=4, scene_center=scene_center)

    # Diagnostic: print camera arrangement info
    cam_positions = poses[:, :3, 3]
    print(f"  Scene center: {scene_center.cpu().tolist()}")
    print(f"  Camera center: {cam_positions.mean(dim=0).cpu().tolist()}")
    for i in range(len(image_paths)):
        p = cam_positions[i].cpu()
        d = (p - scene_center.cpu()).norm().item()
        print(f"  Input cam {i}: pos={p.tolist()}, dist_to_scene={d:.3f}")
    for j in range(4):
        p = novel_c2w[j, :3, 3].cpu()
        d = (p - scene_center.cpu()).norm().item()
        print(f"  Novel cam {j}: pos={p.tolist()}, dist_to_scene={d:.3f}")
    K_avg = intrinsics.mean(dim=0)  # use average intrinsics for novel views

    for j in range(4):
        w2c = torch.linalg.inv(novel_c2w[j]).unsqueeze(0).to(device)
        K = K_avg.unsqueeze(0)

        with torch.no_grad():
            render_colors, render_alphas, _ = rasterization(
                means=means, quats=quats, scales=scales,
                opacities=opacities, colors=colors,
                viewmats=w2c, Ks=K, width=W, height=H,
                sh_degree=3, packed=True,
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
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    cfg = RI3DConfig(scene_dir=Path(args.scene),
                     output_dir=Path(args.output) if args.output else None)
    run_step4(cfg)
