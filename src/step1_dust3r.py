"""Step 1: Camera pose estimation and coarse depth using DUSt3R.

Outputs per scene:
  - outputs/<scene>/dust3r_poses.pt        (N, 4, 4) cam-to-world
  - outputs/<scene>/dust3r_intrinsics.pt   (N, 3, 3)
  - outputs/<scene>/dust3r_depths/         per-view depth maps (.pt and .png)
  - outputs/<scene>/dust3r_confidence/     per-view confidence maps
  - outputs/<scene>/dust3r_pointcloud.ply  combined point cloud
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RI3DConfig


def load_scene_images(scene_dir: Path) -> list[str]:
    """Find all images in a scene directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    paths = sorted(
        p for p in scene_dir.iterdir()
        if p.suffix.lower() in exts
    )
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found in {scene_dir}")
    print(f"Found {len(paths)} images in {scene_dir}")
    return [str(p) for p in paths]


def select_views(image_paths: list[str], n_views: int) -> list[str]:
    """Select n_views well-spaced images from the full set.

    If there are more images than n_views, evenly sample.
    If there are exactly n_views or fewer, return all.
    """
    n = len(image_paths)
    if n <= n_views:
        print(f"Using all {n} images (requested {n_views})")
        return image_paths

    # Evenly spaced selection
    indices = [int(round(i * (n - 1) / (n_views - 1))) for i in range(n_views)]
    selected = [image_paths[i] for i in indices]
    print(f"Selected {n_views} views from {n} images: indices {indices}")
    return selected


def save_depth_vis(depth: np.ndarray, path: Path, title: str = ""):
    """Save a depth map as a colorized PNG."""
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


def save_pointcloud_ply(path: Path, points: np.ndarray, colors: np.ndarray):
    """Save point cloud as PLY. points: (N,3), colors: (N,3) uint8."""
    n = points.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for i in range(n):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def run_dust3r(cfg: RI3DConfig):
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    out_dir = cfg.scene_output_dir()
    depth_dir = out_dir / "dust3r_depths"
    conf_dir = out_dir / "dust3r_confidence"
    depth_dir.mkdir(parents=True, exist_ok=True)
    conf_dir.mkdir(parents=True, exist_ok=True)

    # Load images and select N views
    all_image_paths = load_scene_images(cfg.scene_dir)
    image_paths = select_views(all_image_paths, cfg.n_views)
    n_images = len(image_paths)
    print(f"Loading DUSt3R model...")

    # Load model
    model = AsymmetricCroCo3DStereo.from_pretrained(cfg.dust3r_model)
    model = model.to(cfg.device)

    # Load images at 512 resolution
    images = load_images(image_paths, size=512)
    print(f"Loaded {n_images} images")

    # Create all pairs (for few images, complete graph is fine)
    pairs = make_pairs(images, scene_graph="complete", symmetrize=True)
    print(f"Created {len(pairs)} image pairs")

    # Pairwise inference
    print("Running pairwise inference...")
    output = inference(pairs, model, cfg.device, batch_size=1, verbose=True)

    # Global alignment
    print("Running global alignment...")
    mode = GlobalAlignerMode.PointCloudOptimizer if n_images > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=cfg.device, mode=mode)

    if n_images > 2:
        loss = scene.compute_global_alignment(
            init="mst", niter=300, schedule="cosine", lr=0.01
        )
        print(f"Global alignment loss: {loss:.4f}")

    # Extract results
    intrinsics = scene.get_intrinsics().detach().cpu()       # (N, 3, 3)
    poses = scene.get_im_poses().detach().cpu()              # (N, 4, 4) cam-to-world
    depthmaps = scene.get_depthmaps()                        # list of (H, W)
    pts3d_list = scene.get_pts3d()                           # list of (H, W, 3)
    conf_maps = scene.get_conf()                             # list of (H, W)
    conf_masks = scene.get_masks()                           # list of (H, W) bool
    rgb_imgs = scene.imgs                                    # list of (H, W, 3) numpy

    # Save poses and intrinsics
    torch.save(poses, out_dir / "dust3r_poses.pt")
    torch.save(intrinsics, out_dir / "dust3r_intrinsics.pt")
    print(f"Saved poses {poses.shape} and intrinsics {intrinsics.shape}")

    # Save image paths for later reference
    torch.save(image_paths, out_dir / "image_paths.pt")

    # Save per-view depth, confidence, and visualizations
    all_pts = []
    all_colors = []

    for i in range(n_images):
        depth_np = depthmaps[i].detach().cpu().numpy()
        conf_np = conf_maps[i].detach().cpu().numpy()
        mask_np = conf_masks[i].detach().cpu().numpy()
        pts_np = pts3d_list[i].detach().cpu().numpy()
        rgb_np = rgb_imgs[i]

        # Save tensors
        torch.save(depthmaps[i].detach().cpu(), depth_dir / f"depth_{i:03d}.pt")
        torch.save(conf_maps[i].detach().cpu(), conf_dir / f"conf_{i:03d}.pt")
        torch.save(torch.from_numpy(mask_np), conf_dir / f"mask_{i:03d}.pt")

        # Save visualizations
        name = Path(image_paths[i]).stem
        save_depth_vis(depth_np, depth_dir / f"depth_{i:03d}_{name}.png",
                       f"DUSt3R Depth - View {i}")
        save_depth_vis(conf_np, conf_dir / f"conf_{i:03d}_{name}.png",
                       f"DUSt3R Confidence - View {i}")

        # Collect points for point cloud (high confidence only)
        pts_flat = pts_np[mask_np].reshape(-1, 3)
        # Colors from image (0-1 range to 0-255)
        colors_flat = (rgb_np[mask_np].reshape(-1, 3) * 255).astype(np.uint8)
        all_pts.append(pts_flat)
        all_colors.append(colors_flat)

        print(f"  View {i} ({name}): depth range [{depth_np[mask_np].min():.2f}, "
              f"{depth_np[mask_np].max():.2f}], "
              f"conf mask {mask_np.sum()}/{mask_np.size} pixels")

    # Combine and save point cloud
    all_pts = np.concatenate(all_pts, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    # Subsample if too many points (for manageable PLY file)
    max_pts = 500_000
    if all_pts.shape[0] > max_pts:
        idx = np.random.choice(all_pts.shape[0], max_pts, replace=False)
        all_pts = all_pts[idx]
        all_colors = all_colors[idx]

    ply_path = out_dir / "dust3r_pointcloud.ply"
    save_pointcloud_ply(ply_path, all_pts, all_colors)
    print(f"Saved point cloud: {all_pts.shape[0]} points -> {ply_path}")

    # Print camera info
    for i in range(n_images):
        pos = poses[i, :3, 3].numpy()
        fx = intrinsics[i, 0, 0].item()
        print(f"  Cam {i}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), fx={fx:.1f}")

    # Free model VRAM
    del model, scene
    torch.cuda.empty_cache()

    print(f"\nStep 1 complete! Outputs in {out_dir}")
    return {
        "poses": poses,
        "intrinsics": intrinsics,
        "image_paths": image_paths,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: DUSt3R camera + depth estimation")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene image directory")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--n_views", type=int, default=3, help="Number of input views to select")
    args = parser.parse_args()

    cfg = RI3DConfig(scene_dir=Path(args.scene), output_dir=Path(args.output),
                     n_views=args.n_views)
    run_dust3r(cfg)
