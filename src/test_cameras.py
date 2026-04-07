"""Test script for novel camera generation. Visualizes camera positions in 3D."""
import torch
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def generate_elliptical_cameras(poses, n_cameras, scene_center=None):
    """Current implementation to test and fix."""
    positions = poses[:, :3, 3]
    cam_center = positions.mean(dim=0)

    forwards = poses[:, :3, 2]
    mean_forward = forwards.mean(dim=0)
    mean_forward = mean_forward / mean_forward.norm()

    if scene_center is None:
        cam_spread = (positions - cam_center).norm(dim=1).mean().item()
        depth_est = max(cam_spread * 2.0, 0.5)
        scene_center = cam_center + depth_est * mean_forward

    mean_up = -poses[:, :3, 1].mean(dim=0)
    mean_up = mean_up / mean_up.norm()

    look_dir = scene_center - cam_center
    look_horiz = look_dir - (look_dir @ mean_up) * mean_up
    if look_horiz.norm() < 1e-6:
        look_horiz = mean_forward - (mean_forward @ mean_up) * mean_up
    look_horiz = look_horiz / look_horiz.norm()

    axis1 = torch.linalg.cross(mean_up, look_horiz)
    axis1 = axis1 / axis1.norm()
    axis2 = look_horiz

    sc_offsets = positions - scene_center.unsqueeze(0)
    horiz_offsets = sc_offsets - (sc_offsets @ mean_up).unsqueeze(-1) * mean_up.unsqueeze(0)
    proj1 = (horiz_offsets @ axis1).abs()
    proj2 = (horiz_offsets @ axis2).abs()
    radius1 = max(proj1.mean().item() * 1.1, 0.3)
    radius2 = max(proj2.mean().item() * 1.1, 0.3)

    mean_height = (sc_offsets @ mean_up).median().item()

    angles = torch.linspace(0, 2 * math.pi, n_cameras + 1)[:-1]
    novel_c2w = torch.zeros(n_cameras, 4, 4)

    for i, angle in enumerate(angles):
        pos = (scene_center
               + radius1 * torch.cos(angle) * axis1
               + radius2 * torch.sin(angle) * axis2
               + mean_height * mean_up)

        fwd = scene_center - pos
        fwd = fwd / fwd.norm()

        r_vec = torch.linalg.cross(fwd, mean_up)
        if r_vec.norm() < 1e-6:
            r_vec = axis1
        r_vec = r_vec / r_vec.norm()
        down = torch.linalg.cross(fwd, r_vec)
        if down @ (-mean_up) < 0:
            down = -down
            r_vec = -r_vec

        novel_c2w[i, :3, 0] = r_vec
        novel_c2w[i, :3, 1] = down
        novel_c2w[i, :3, 2] = fwd
        novel_c2w[i, :3, 3] = pos
        novel_c2w[i, 3, 3] = 1.0

    return novel_c2w


def plot_cameras(input_poses, novel_c2w, scene_center, title, filename, mean_up=None):
    """Plot input and novel cameras: 3D + top-down + side views."""
    fig = plt.figure(figsize=(20, 14))
    ax = fig.add_subplot(221, projection='3d')

    # Input cameras
    for i in range(len(input_poses)):
        p = input_poses[i, :3, 3].numpy()
        fwd = input_poses[i, :3, 2].numpy() * 0.05  # +Z = forward in OpenCV
        up = -input_poses[i, :3, 1].numpy() * 0.03
        ax.scatter(*p, c='blue', s=100, marker='o', zorder=5)
        ax.quiver(*p, *fwd, color='blue', arrow_length_ratio=0.3, linewidth=2)
        ax.quiver(*p, *up, color='cyan', arrow_length_ratio=0.3, linewidth=1)
        ax.text(p[0], p[1], p[2], f'  in{i}', fontsize=9, color='blue')

    # Novel cameras
    for i in range(len(novel_c2w)):
        p = novel_c2w[i, :3, 3].numpy()
        fwd = novel_c2w[i, :3, 2].numpy() * 0.05
        ax.scatter(*p, c='red', s=80, marker='^', zorder=5)
        ax.quiver(*p, *fwd, color='red', arrow_length_ratio=0.3, linewidth=2)
        ax.text(p[0], p[1], p[2], f'  n{i}', fontsize=9, color='red')

    # Scene center
    sc = scene_center.numpy()
    ax.scatter(*sc, c='green', s=200, marker='*', zorder=10)
    ax.text(sc[0], sc[1], sc[2], '  scene', fontsize=10, color='green')

    # Camera center
    cam_center = input_poses[:, :3, 3].mean(dim=0).numpy()
    ax.scatter(*cam_center, c='purple', s=100, marker='D')
    ax.text(cam_center[0], cam_center[1], cam_center[2], '  cam_ctr', fontsize=9, color='purple')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Equal aspect ratio
    all_pts = np.vstack([
        input_poses[:, :3, 3].numpy(),
        novel_c2w[:, :3, 3].numpy(),
        sc.reshape(1, 3),
    ])
    mid = all_pts.mean(axis=0)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # Top-down view (looking along mean_up direction)
    if mean_up is not None:
        up = mean_up.numpy()
        # Project all points onto plane perpendicular to up
        # Use axis1 = right, axis2 = forward in this plane
        fwd_proj = sc - cam_center  # cam->scene direction
        fwd_proj = fwd_proj - np.dot(fwd_proj, up) * up
        fwd_proj = fwd_proj / (np.linalg.norm(fwd_proj) + 1e-8)
        right_proj = np.cross(up, fwd_proj)
        right_proj = right_proj / (np.linalg.norm(right_proj) + 1e-8)

        ax2 = fig.add_subplot(222)
        ax2.set_title("Top-down view (looking along up)")
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)

        for i in range(len(input_poses)):
            p = input_poses[i, :3, 3].numpy()
            x = np.dot(p, right_proj)
            y = np.dot(p, fwd_proj)
            ax2.plot(x, y, 'bo', markersize=10)
            ax2.annotate(f'in{i}', (x, y), fontsize=8, color='blue')

        for i in range(len(novel_c2w)):
            p = novel_c2w[i, :3, 3].numpy()
            x = np.dot(p, right_proj)
            y = np.dot(p, fwd_proj)
            ax2.plot(x, y, 'r^', markersize=8)
            ax2.annotate(f'n{i}', (x, y), fontsize=8, color='red')

        sx = np.dot(sc, right_proj)
        sy = np.dot(sc, fwd_proj)
        ax2.plot(sx, sy, 'g*', markersize=15)
        ax2.annotate('scene', (sx, sy), fontsize=9, color='green')
        ax2.set_xlabel('Right')
        ax2.set_ylabel('Forward (cam->scene)')

        # Side view: height along up vs distance in forward direction
        ax3 = fig.add_subplot(223)
        ax3.set_title("Side view (height vs forward)")
        ax3.grid(True, alpha=0.3)

        for i in range(len(input_poses)):
            p = input_poses[i, :3, 3].numpy()
            h = np.dot(p, up)
            f = np.dot(p, fwd_proj)
            ax3.plot(f, h, 'bo', markersize=10)
            ax3.annotate(f'in{i}', (f, h), fontsize=8, color='blue')

        for i in range(len(novel_c2w)):
            p = novel_c2w[i, :3, 3].numpy()
            h = np.dot(p, up)
            f = np.dot(p, fwd_proj)
            ax3.plot(f, h, 'r^', markersize=8)
            ax3.annotate(f'n{i}', (f, h), fontsize=8, color='red')

        sh = np.dot(sc, up)
        sf = np.dot(sc, fwd_proj)
        ax3.plot(sf, sh, 'g*', markersize=15)
        ax3.annotate('scene', (sf, sh), fontsize=9, color='green')
        ax3.set_xlabel('Forward')
        ax3.set_ylabel('Height (along up)')
        ax3.axhline(y=sh, color='green', linestyle='--', alpha=0.3, label='scene height')
        ax3.legend()

        # Side view: height vs right
        ax4 = fig.add_subplot(224)
        ax4.set_title("Side view (height vs right)")
        ax4.grid(True, alpha=0.3)

        for i in range(len(input_poses)):
            p = input_poses[i, :3, 3].numpy()
            h = np.dot(p, up)
            r = np.dot(p, right_proj)
            ax4.plot(r, h, 'bo', markersize=10)
            ax4.annotate(f'in{i}', (r, h), fontsize=8, color='blue')

        for i in range(len(novel_c2w)):
            p = novel_c2w[i, :3, 3].numpy()
            h = np.dot(p, up)
            r = np.dot(p, right_proj)
            ax4.plot(r, h, 'r^', markersize=8)
            ax4.annotate(f'n{i}', (r, h), fontsize=8, color='red')

        sh2 = np.dot(sc, up)
        sr = np.dot(sc, right_proj)
        ax4.plot(sr, sh2, 'g*', markersize=15)
        ax4.set_xlabel('Right')
        ax4.set_ylabel('Height (along up)')
        ax4.axhline(y=sh2, color='green', linestyle='--', alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")


def load_scene_data(scene_name):
    """Load poses and compute scene center from actual scene data."""
    from pathlib import Path
    out_dir = Path(f"outputs/{scene_name}")

    poses = torch.load(out_dir / "dust3r_poses.pt", weights_only=True).float()
    gaussians = torch.load(out_dir / "init_gaussians.pt", weights_only=True)

    from step4_gaussian_init import compute_scene_center
    scene_center = compute_scene_center(poses, gaussians["means"])

    return poses, scene_center


def test_scene(scene_name):
    """Test camera generation for a scene."""
    print(f"\n=== Testing {scene_name} ===")
    poses, scene_center = load_scene_data(scene_name)

    positions = poses[:, :3, 3]
    cam_center = positions.mean(dim=0)
    mean_up = -poses[:, :3, 1].mean(dim=0)
    mean_up = mean_up / mean_up.norm()

    print(f"Scene center: {scene_center.tolist()}")
    print(f"Camera center: {cam_center.tolist()}")
    print(f"Mean up: {mean_up.tolist()}")

    for i in range(len(poses)):
        p = positions[i]
        # Height = component along mean_up, relative to scene center
        height = ((p - scene_center) @ mean_up).item()
        print(f"Input cam {i}: pos={p.tolist()}, height_above_scene={height:.4f}")

    from step4_gaussian_init import generate_elliptical_cameras as gen_cams
    novel_c2w = gen_cams(poses, 8, scene_center)

    for j in range(len(novel_c2w)):
        p = novel_c2w[j, :3, 3]
        height = ((p - scene_center) @ mean_up).item()
        dist = (p - scene_center).norm().item()
        print(f"Novel cam {j}: pos={p.tolist()}, height={height:.4f}, dist={dist:.3f}")

    plot_cameras(poses, novel_c2w, scene_center,
                 f"{scene_name} — Camera Layout",
                 f"test_cameras_{scene_name}.png",
                 mean_up=mean_up)


if __name__ == "__main__":
    import sys
    scenes = sys.argv[1:] if len(sys.argv) > 1 else ["garden", "bonsai"]
    for scene in scenes:
        test_scene(scene)
