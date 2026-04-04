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
    return [str(p.resolve()) for p in paths]


def _resolve_names(names: list[str], image_paths: list[str]) -> list[str]:
    """Resolve a list of filenames/stems to full paths from image_paths."""
    available = {Path(p).name: p for p in image_paths}
    selected = []
    for name in names:
        name = name.strip()
        if not name:
            continue
        if name in available:
            selected.append(available[name])
        else:
            stem = Path(name).stem
            matches = [p for p in image_paths if Path(p).stem == stem]
            if matches:
                selected.append(matches[0])
            else:
                raise FileNotFoundError(
                    f"View '{name}' not found in {list(available.keys())}"
                )
    return selected


def select_views(image_paths: list[str], n_views, scene_dir: Path = None) -> list[str]:
    """Select views from the full set.

    Priority:
      1. views.txt in scene_dir (one filename per line) — used for multi-scene prep
      2. n_views as comma-separated filenames — used for single-scene CLI
      3. n_views as int — evenly sample that many views (fallback)
    """
    # 1. Check for views.txt in scene directory
    if scene_dir is not None:
        views_file = Path(scene_dir) / "views.txt"
        if views_file.exists():
            names = views_file.read_text().strip().splitlines()
            names = [n.strip() for n in names if n.strip()]
            selected = _resolve_names(names, image_paths)
            print(f"Selected {len(selected)} views from views.txt: "
                  f"{[Path(p).name for p in selected]}")
            return selected

    # 2. Comma-separated filenames from CLI
    if isinstance(n_views, str) and not n_views.isdigit():
        names = n_views.split(",")
        selected = _resolve_names(names, image_paths)
        print(f"Selected {len(selected)} specific views: "
              f"{[Path(p).name for p in selected]}")
        return selected

    # 3. Evenly spaced selection by count
    n_views = int(n_views)
    n = len(image_paths)
    if n <= n_views:
        print(f"Using all {n} images (requested {n_views})")
        return image_paths

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


def compute_triangulation_quality(poses, conf_masks):
    """Compute a triangulation quality score for the input views.

    Returns a dict with per-pair and overall metrics:
      - baselines: pairwise camera distances
      - angular_divs: pairwise angular differences between viewing directions
      - confidence_coverage: fraction of high-confidence pixels per view
      - score: overall quality 0-100 (higher = better triangulation)

    Scoring:
      - Baseline diversity: wider spread of camera positions is better
      - Angular diversity: more varied viewing angles give better triangulation
      - Confidence: higher DUSt3R confidence means more reliable depth
    """
    n = len(poses)
    positions = poses[:, :3, 3]
    forwards = poses[:, :3, 2]

    baselines = []
    angular_divs = []
    for i in range(n):
        for j in range(i + 1, n):
            bl = (positions[i] - positions[j]).norm().item()
            baselines.append(bl)
            cos_a = (forwards[i] @ forwards[j]).item()
            ang = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
            angular_divs.append(ang)

    conf_coverages = []
    for mask in conf_masks:
        mask_np = mask.detach().cpu().numpy() if torch.is_tensor(mask) else mask
        conf_coverages.append(mask_np.sum() / mask_np.size)

    # --- Score components (each 0-1) ---

    # Baseline score: ratio of min to max baseline
    # Perfect = 1.0 (all baselines equal), bad = 0.0 (one pair nearly co-located)
    bl_arr = np.array(baselines)
    if bl_arr.max() > 1e-8:
        baseline_score = bl_arr.min() / bl_arr.max()
    else:
        baseline_score = 0.0

    # Angular diversity score: want angles spread between 30-120 degrees
    # Too small (<15°) = nearly same view, too large (>150°) = opposing views
    ang_arr = np.array(angular_divs)
    ang_scores = []
    for a in ang_arr:
        if a < 15:
            ang_scores.append(a / 15.0 * 0.3)  # very poor
        elif a < 30:
            ang_scores.append(0.3 + (a - 15) / 15.0 * 0.4)
        elif a <= 120:
            ang_scores.append(0.7 + 0.3 * (1.0 - abs(a - 75) / 45.0))  # peak at 75°
        elif a <= 150:
            ang_scores.append(0.7 - (a - 120) / 30.0 * 0.3)
        else:
            ang_scores.append(0.4)  # opposing views still usable
    angular_score = np.mean(ang_scores)

    # Confidence score: average coverage
    conf_score = np.mean(conf_coverages)

    # Overall: weighted combination
    overall = 0.35 * baseline_score + 0.45 * angular_score + 0.20 * conf_score
    score = int(round(overall * 100))

    return {
        "baselines": baselines,
        "angular_divs": angular_divs,
        "confidence_coverages": conf_coverages,
        "baseline_score": baseline_score,
        "angular_score": angular_score,
        "confidence_score": conf_score,
        "score": score,
    }


def print_quality_report(quality: dict, image_paths: list[str], poses):
    """Print a human-readable triangulation quality report."""
    n = len(image_paths)
    names = [Path(p).stem for p in image_paths]

    print(f"\n{'='*60}")
    print(f"  TRIANGULATION QUALITY REPORT")
    print(f"{'='*60}")

    # Pairwise metrics
    idx = 0
    print(f"\n  Pairwise baselines and angular differences:")
    for i in range(n):
        for j in range(i + 1, n):
            bl = quality["baselines"][idx]
            ang = quality["angular_divs"][idx]
            status = ""
            if ang < 15:
                status = " << POOR: nearly identical views"
            elif bl < 0.05:
                status = " << POOR: cameras too close"
            elif ang > 150:
                status = " (opposing views)"
            print(f"    {names[i]} <-> {names[j]}: "
                  f"baseline={bl:.4f}, angle={ang:.1f}°{status}")
            idx += 1

    # Per-view confidence
    print(f"\n  Per-view DUSt3R confidence coverage:")
    for i in range(n):
        cov = quality["confidence_coverages"][i] * 100
        print(f"    {names[i]}: {cov:.1f}%")

    # Component scores
    print(f"\n  Score breakdown:")
    print(f"    Baseline diversity:  {quality['baseline_score']*100:.0f}/100")
    print(f"    Angular diversity:   {quality['angular_score']*100:.0f}/100")
    print(f"    Confidence coverage: {quality['confidence_score']*100:.0f}/100")

    score = quality["score"]
    if score >= 70:
        grade = "GOOD"
    elif score >= 45:
        grade = "FAIR"
    else:
        grade = "POOR — consider choosing different input views"

    print(f"\n  Overall triangulation quality: {score}/100 ({grade})")
    print(f"{'='*60}\n")


def run_dust3r(cfg: RI3DConfig, model=None):
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

    # Load images and select views
    all_image_paths = load_scene_images(cfg.scene_dir)
    image_paths = select_views(all_image_paths, cfg.n_views, cfg.scene_dir)
    n_images = len(image_paths)

    # Load model if not provided externally
    _owns_model = model is None
    if _owns_model:
        print(f"Loading DUSt3R model...")
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
            init="mst", niter=512, schedule="cosine", lr=0.01
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

    # Triangulation quality report
    quality = compute_triangulation_quality(poses, conf_masks)
    print_quality_report(quality, image_paths, poses)

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

    # Free model VRAM only if we loaded it ourselves
    del scene
    if _owns_model:
        del model
    torch.cuda.empty_cache()

    print(f"\nStep 1 complete! Outputs in {out_dir}")
    return {
        "poses": poses,
        "intrinsics": intrinsics,
        "image_paths": image_paths,
        "quality": quality,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: DUSt3R camera + depth estimation")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene image directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--n_views", type=str, default="3",
                        help="Number of views to select (int), or comma-separated "
                             "filenames e.g. 'DSC_001.jpg,DSC_005.jpg,DSC_010.jpg'")
    args = parser.parse_args()

    cfg = RI3DConfig(scene_dir=Path(args.scene),
                     output_dir=Path(args.output) if args.output else None,
                     n_views=args.n_views)
    run_dust3r(cfg)
