"""Extract all training pairs from an existing training_pairs.pt into
paginated PNG previews. Does NOT retrain.

Usage:
    python src/extract_pairs.py --scene output/bonsai
    python src/extract_pairs.py --pairs output/bonsai/repair_training_data/training_pairs.pt
"""
import argparse
from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract(pairs_path: Path, out_dir: Path, per_page: int = 8,
            scene_name: str = "") -> None:
    pairs = torch.load(pairs_path, weights_only=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(pairs)
    n_pages = (n_total + per_page - 1) // per_page
    for page in range(n_pages):
        start = page * per_page
        end = min(start + per_page, n_total)
        n_show = end - start
        fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
        if n_show == 1:
            axes = axes.reshape(-1, 1)
        for col, j in enumerate(range(start, end)):
            axes[0, col].imshow(pairs[j][0].numpy())
            axes[0, col].set_title(f"Corrupted (pair {j})")
            axes[0, col].axis("off")
            axes[1, col].imshow(pairs[j][1].numpy())
            axes[1, col].set_title(f"Clean (pair {j})")
            axes[1, col].axis("off")
        fig.suptitle(
            f"Leave-One-Out Pairs — {scene_name} "
            f"(page {page + 1}/{n_pages}, pairs {start}-{end - 1} of {n_total})"
        )
        out = out_dir / f"pairs_preview_{page:03d}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")
    print(f"Done: {n_total} pairs across {n_pages} page(s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str,
                    help="Path to training_pairs.pt")
    ap.add_argument("--scene", type=str,
                    help="Scene output dir (will use <scene>/repair_training_data/training_pairs.pt)")
    ap.add_argument("--per_page", type=int, default=8)
    args = ap.parse_args()

    if args.pairs:
        pairs_path = Path(args.pairs)
        out_dir = pairs_path.parent
        scene_name = pairs_path.parent.parent.name
    elif args.scene:
        scene_dir = Path(args.scene)
        pairs_path = scene_dir / "repair_training_data" / "training_pairs.pt"
        out_dir = pairs_path.parent
        scene_name = scene_dir.name
    else:
        ap.error("pass --pairs or --scene")

    extract(pairs_path, out_dir, per_page=args.per_page, scene_name=scene_name)
