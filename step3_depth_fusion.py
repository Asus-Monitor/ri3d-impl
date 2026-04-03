"""Step 3: Poisson depth fusion — combine DUSt3R + monocular depth (Eq. 2).

The objective:
  d* = argmin_d [ M ⊙ ||d - d_D||² + λ ||∇d - ∇d_M||² ]

Where:
  d_D = DUSt3R depth (3D-consistent but smooth)
  d_M = aligned monocular depth (detailed but relative)
  M   = high-confidence mask from DUSt3R

Steps:
  1. Align monocular depth to DUSt3R scale (piecewise linear in high-conf regions)
  2. Solve Poisson system (sparse linear solve)
  3. Apply bilateral filter for sharp edges

Outputs:
  - outputs/<scene>/fused_depths/  per-view fused depth maps (.pt and .png)
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import cv2
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RI3DConfig


def save_depth_vis(depth: np.ndarray, path: Path, title: str = ""):
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


def save_comparison(depths: dict[str, np.ndarray], path: Path, suptitle: str = ""):
    """Save side-by-side depth comparison."""
    n = len(depths)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, d) in zip(axes, depths.items()):
        valid = d[d > 0] if np.any(d > 0) else d.ravel()
        vmin, vmax = np.percentile(valid, [2, 98])
        ax.imshow(d, cmap="turbo", vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(suptitle)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def align_mono_to_dust3r(mono: np.ndarray, dust3r: np.ndarray, mask: np.ndarray,
                          n_bins: int = 10) -> np.ndarray:
    """Align monocular depth to DUSt3R scale using piecewise linear mapping.

    Only uses high-confidence regions (mask=True) for fitting.
    Linearly extrapolates for out-of-range values.
    """
    mono_valid = mono[mask]
    dust3r_valid = dust3r[mask]

    if len(mono_valid) < 100:
        # Fallback: simple linear fit
        scale = dust3r_valid.std() / (mono_valid.std() + 1e-8)
        shift = dust3r_valid.mean() - scale * mono_valid.mean()
        return mono * scale + shift

    # Compute percentile bins on mono depth in masked region
    percentiles = np.linspace(0, 100, n_bins + 1)
    mono_edges = np.percentile(mono_valid, percentiles)

    # For each bin, compute median mono and median dust3r depth
    mono_centers = []
    dust3r_centers = []
    for j in range(n_bins):
        lo, hi = mono_edges[j], mono_edges[j + 1]
        if j == n_bins - 1:
            in_bin = (mono_valid >= lo) & (mono_valid <= hi)
        else:
            in_bin = (mono_valid >= lo) & (mono_valid < hi)
        if in_bin.sum() > 0:
            mono_centers.append(np.median(mono_valid[in_bin]))
            dust3r_centers.append(np.median(dust3r_valid[in_bin]))

    mono_centers = np.array(mono_centers)
    dust3r_centers = np.array(dust3r_centers)

    # Piecewise linear interpolation with extrapolation
    aligned = np.interp(mono.ravel(), mono_centers, dust3r_centers).reshape(mono.shape)
    return aligned


def solve_poisson_fusion(dust3r: np.ndarray, mono_aligned: np.ndarray,
                          mask: np.ndarray, lam: float = 10.0) -> np.ndarray:
    """Solve the Poisson fusion objective (Eq. 2 from paper).

    d* = argmin_d [ M ⊙ ||d - d_D||² + λ ||∇d - ∇d_M||² ]

    Setting gradient to zero gives a sparse linear system:
      (M + λ L) d = M ⊙ d_D + λ div(∇d_M)
    where L is the discrete Laplacian.
    """
    H, W = dust3r.shape
    N = H * W

    # Index mapping
    def idx(r, c):
        return r * W + c

    # Build sparse Laplacian and divergence of mono gradient
    # Using 4-connected neighbors
    rows, cols, vals = [], [], []
    rhs = np.zeros(N)

    for r in range(H):
        for c in range(W):
            i = idx(r, c)
            n_neighbors = 0

            # Data term: M(i) * (d(i) - d_D(i)) contributes M(i) on diagonal, M(i)*d_D(i) to rhs
            if mask[r, c]:
                rows.append(i); cols.append(i); vals.append(1.0)
                rhs[i] += dust3r[r, c]

            # Gradient term: λ * sum_neighbors (d(i) - d(j) - (d_M(i) - d_M(j)))²
            # Contributes λ on diagonal per neighbor, -λ off-diagonal
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    j = idx(nr, nc)
                    n_neighbors += 1
                    rows.append(i); cols.append(i); vals.append(lam)
                    rows.append(i); cols.append(j); vals.append(-lam)
                    # RHS: λ * (d_M(i) - d_M(j))
                    rhs[i] += lam * (mono_aligned[r, c] - mono_aligned[nr, nc])

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))

    print(f"    Solving {N}x{N} sparse system...")
    d_flat = spsolve(A, rhs)
    return d_flat.reshape(H, W)


def solve_poisson_fusion_fast(dust3r: np.ndarray, mono_aligned: np.ndarray,
                               mask: np.ndarray, lam: float = 10.0) -> np.ndarray:
    """Faster Poisson fusion using vectorized sparse matrix construction."""
    H, W = dust3r.shape
    N = H * W

    # Pixel indices
    idx = np.arange(N).reshape(H, W)

    # Lists for sparse entries
    row_list, col_list, val_list = [], [], []
    rhs = np.zeros(N)

    # Data term: M ⊙ ||d - d_D||²
    mask_flat = mask.ravel().astype(np.float64)
    diag_data = np.where(mask.ravel(), 1.0, 0.0)
    row_list.append(np.arange(N))
    col_list.append(np.arange(N))
    val_list.append(diag_data)
    rhs += mask_flat * dust3r.ravel()

    # Gradient term: λ ||∇d - ∇d_M||²
    # For each direction, add entries for (d_i - d_j - (dM_i - dM_j))²
    shifts = [
        (slice(None), slice(None, -1), slice(None), slice(1, None)),  # right
        (slice(None, -1), slice(None), slice(1, None), slice(None)),  # down
    ]

    for r1, c1, r2, c2 in shifts:
        i = idx[r1, c1].ravel()
        j = idx[r2, c2].ravel()
        n = len(i)

        # A[i,i] += lam, A[j,j] += lam, A[i,j] -= lam, A[j,i] -= lam
        row_list.append(i); col_list.append(i); val_list.append(np.full(n, lam))
        row_list.append(j); col_list.append(j); val_list.append(np.full(n, lam))
        row_list.append(i); col_list.append(j); val_list.append(np.full(n, -lam))
        row_list.append(j); col_list.append(i); val_list.append(np.full(n, -lam))

        # RHS contribution
        grad_mono = mono_aligned[r1, c1].ravel() - mono_aligned[r2, c2].ravel()
        rhs[i] += lam * grad_mono
        rhs[j] -= lam * grad_mono

    rows = np.concatenate(row_list)
    cols_arr = np.concatenate(col_list)
    vals = np.concatenate(val_list)

    A = sparse.csr_matrix((vals, (rows, cols_arr)), shape=(N, N))

    print(f"    Solving {H}x{W} sparse system ({A.nnz} nonzeros)...")
    d_flat = spsolve(A, rhs)
    return d_flat.reshape(H, W).astype(np.float32)


def run_depth_fusion(cfg: RI3DConfig):
    out_dir = cfg.scene_output_dir()
    fused_dir = out_dir / "fused_depths"
    fused_dir.mkdir(parents=True, exist_ok=True)

    image_paths = torch.load(out_dir / "image_paths.pt", weights_only=False)
    n_images = len(image_paths)

    print(f"Fusing depth maps for {n_images} views (λ={cfg.poisson_lambda})...")

    for i in range(n_images):
        name = Path(image_paths[i]).stem
        print(f"  View {i} ({name}):")

        # Load DUSt3R depth and confidence
        dust3r_depth = torch.load(out_dir / "dust3r_depths" / f"depth_{i:03d}.pt",
                                   weights_only=True).numpy().astype(np.float64)
        conf = torch.load(out_dir / "dust3r_confidence" / f"conf_{i:03d}.pt",
                           weights_only=True).numpy()
        mask = conf > cfg.dust3r_confidence_threshold  # high-confidence mask

        # Load monocular depth
        mono_depth = torch.load(out_dir / "mono_depths" / f"mono_depth_{i:03d}.pt",
                                 weights_only=True).numpy().astype(np.float64)

        print(f"    DUSt3R range: [{dust3r_depth.min():.3f}, {dust3r_depth.max():.3f}]")
        print(f"    Mono range:   [{mono_depth.min():.3f}, {mono_depth.max():.3f}]")
        print(f"    High-conf mask: {mask.sum()}/{mask.size} pixels ({100*mask.mean():.1f}%)")

        # Step 1: Align monocular depth to DUSt3R scale
        mono_aligned = align_mono_to_dust3r(mono_depth, dust3r_depth, mask)
        print(f"    Aligned mono range: [{mono_aligned.min():.3f}, {mono_aligned.max():.3f}]")

        # Step 2: Solve Poisson fusion
        fused = solve_poisson_fusion_fast(dust3r_depth, mono_aligned, mask, cfg.poisson_lambda)

        # Step 3: Bilateral filter for sharp edges
        # Normalize to 0-1 range for bilateral filter, then scale back
        fmin, fmax = fused.min(), fused.max()
        fused_norm = ((fused - fmin) / (fmax - fmin + 1e-8) * 255).astype(np.uint8)
        fused_bilateral = cv2.bilateralFilter(
            fused_norm, cfg.bilateral_d, cfg.bilateral_sigma_color, cfg.bilateral_sigma_space
        )
        fused_final = fused_bilateral.astype(np.float32) / 255.0 * (fmax - fmin) + fmin

        # Save
        torch.save(torch.from_numpy(fused_final), fused_dir / f"fused_depth_{i:03d}.pt")

        # Save comparison visualization
        save_comparison(
            {
                "DUSt3R": dust3r_depth.astype(np.float32),
                "Mono (aligned)": mono_aligned.astype(np.float32),
                "Fused": fused.astype(np.float32),
                "Fused + Bilateral": fused_final,
            },
            fused_dir / f"comparison_{i:03d}_{name}.png",
            f"Depth Fusion - View {i} ({name})"
        )

        print(f"    Fused range: [{fused_final.min():.3f}, {fused_final.max():.3f}]")

    print(f"\nStep 3 complete! Fused depths in {fused_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Poisson depth fusion")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene image directory")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    cfg = RI3DConfig(scene_dir=Path(args.scene), output_dir=Path(args.output))
    run_depth_fusion(cfg)
