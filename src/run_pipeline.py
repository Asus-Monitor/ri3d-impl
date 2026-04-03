"""RI3D Full Pipeline Runner.

Workflow:
  1. Prep all scenes (steps 1-4 per scene):
       python run_pipeline.py --dataset dataset --n_views 3 --prep

  2. Train shared models (steps 5+7 on all scenes combined):
       python run_pipeline.py --dataset dataset --train_models

  3. Optimize a specific scene (steps 6+8):
       python run_pipeline.py --scene dataset/garden --optimize

  OR run everything for a single scene (uses all scenes for model training):
       python run_pipeline.py --scene dataset/garden --dataset dataset --n_views 3

  OR run only a specific step:
       python run_pipeline.py --scene dataset/garden --step 3 --only
"""
import argparse
from pathlib import Path

from config import RI3DConfig


def run_prep_scene(cfg: RI3DConfig):
    """Run steps 1-4 for a single scene."""
    from step1_dust3r import run_dust3r
    from step2_mono_depth import run_mono_depth
    from step3_depth_fusion import run_depth_fusion
    from step4_gaussian_init import run_step4

    print(f"\n{'='*60}")
    print(f"  Preparing scene: {cfg.scene_name}")
    print(f"{'='*60}\n")

    print("--- Step 1: DUSt3R ---")
    run_dust3r(cfg)

    print("\n--- Step 2: Monocular Depth ---")
    run_mono_depth(cfg)

    print("\n--- Step 3: Depth Fusion ---")
    run_depth_fusion(cfg)

    print("\n--- Step 4: Gaussian Init ---")
    run_step4(cfg)


def run_prep_all_scenes(cfg: RI3DConfig):
    """Run steps 1-4 for ALL scenes in the dataset."""
    scenes = cfg.list_scenes()
    print(f"Preparing {len(scenes)} scenes...")

    for scene_dir in scenes:
        scene_cfg = RI3DConfig(
            scene_dir=scene_dir, dataset_dir=cfg.dataset_dir,
            output_dir=cfg.output_dir, n_views=cfg.n_views,
            device=cfg.device, dtype=cfg.dtype,
        )
        # Skip if already done
        if (scene_cfg.scene_output_dir() / "init_gaussians.pt").exists():
            print(f"  Skipping {scene_dir.name}: already prepared")
            continue
        run_prep_scene(scene_cfg)


def run_train_models(cfg: RI3DConfig):
    """Train shared repair + inpainting models on all scenes."""
    from step5_repair_model import run_step5
    from step7_inpainting_model import run_step7

    print(f"\n{'='*60}")
    print(f"  Training shared models on all scenes")
    print(f"{'='*60}\n")

    print("--- Step 5: Repair Model ---")
    run_step5(cfg)

    print("\n--- Step 7: Inpainting Model ---")
    run_step7(cfg)


def run_optimize_scene(cfg: RI3DConfig):
    """Run steps 6+8 (optimization) for a single scene."""
    from step6_stage1_optim import run_stage1
    from step8_stage2_optim import run_stage2

    print(f"\n{'='*60}")
    print(f"  Optimizing scene: {cfg.scene_name}")
    print(f"{'='*60}\n")

    print("--- Step 6: Stage 1 ---")
    run_stage1(cfg)

    print("\n--- Step 8: Stage 2 ---")
    run_stage2(cfg)


def run_single_step(step_num: int, cfg: RI3DConfig):
    """Run a single pipeline step."""
    STEPS = {
        1: ("DUSt3R: camera poses + coarse depth", "step1_dust3r", "run_dust3r"),
        2: ("Depth Anything V2: monocular depth", "step2_mono_depth", "run_mono_depth"),
        3: ("Poisson depth fusion", "step3_depth_fusion", "run_depth_fusion"),
        4: ("Gaussian initialization + render", "step4_gaussian_init", "run_step4"),
        5: ("Repair model training (all scenes)", "step5_repair_model", "run_step5"),
        6: ("Stage 1 optimization", "step6_stage1_optim", "run_stage1"),
        7: ("Inpainting model training (all scenes)", "step7_inpainting_model", "run_step7"),
        8: ("Stage 2 optimization", "step8_stage2_optim", "run_stage2"),
    }

    name, module_name, func_name = STEPS[step_num]
    print(f"\n{'='*60}")
    print(f"  Step {step_num}: {name}")
    print(f"{'='*60}\n")

    module = __import__(module_name)
    func = getattr(module, func_name)
    func(cfg)


def main():
    parser = argparse.ArgumentParser(description="RI3D Pipeline")
    parser.add_argument("--scene", type=str, default=None, help="Path to a single scene directory")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset root (all scenes)")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--n_views", type=int, default=3, help="Number of input views per scene")

    # Mode flags
    parser.add_argument("--prep", action="store_true", help="Run steps 1-4 for all scenes")
    parser.add_argument("--train_models", action="store_true", help="Train shared models (steps 5+7)")
    parser.add_argument("--optimize", action="store_true", help="Run optimization (steps 6+8) for --scene")

    # Single step mode
    parser.add_argument("--step", type=int, default=None, help="Run a specific step (1-8)")
    parser.add_argument("--only", action="store_true", help="Run only the specified --step")

    args = parser.parse_args()

    # Build config
    scene_dir = Path(args.scene) if args.scene else Path(args.dataset)
    cfg = RI3DConfig(
        scene_dir=scene_dir,
        dataset_dir=Path(args.dataset),
        output_dir=Path(args.output),
        n_views=args.n_views,
    )

    print(f"RI3D Pipeline")
    print(f"  Scene: {args.scene or '(all scenes)'}")
    print(f"  Dataset: {cfg.dataset_dir}")
    print(f"  N views: {cfg.n_views}")
    print(f"  Output: {cfg.output_dir}")
    print(f"  Device: {cfg.device}, dtype: {cfg.dtype}")

    # Single step mode
    if args.step is not None:
        if args.scene is None and args.step not in (5, 7):
            parser.error("--scene is required for this step")
        run_single_step(args.step, cfg)
        return

    # Prep mode: steps 1-4 for all scenes
    if args.prep:
        run_prep_all_scenes(cfg)
        return

    # Train mode: steps 5+7 on all scenes
    if args.train_models:
        run_train_models(cfg)
        return

    # Optimize mode: steps 6+8 for one scene
    if args.optimize:
        if args.scene is None:
            parser.error("--scene is required for --optimize")
        run_optimize_scene(cfg)
        return

    # Default: full pipeline for one scene (using all scenes for model training)
    if args.scene is None:
        parser.error("Provide --scene, or use --prep / --train_models / --optimize")

    # Full pipeline
    run_prep_all_scenes(cfg)
    run_train_models(cfg)
    run_optimize_scene(cfg)

    ckpt = cfg.scene_output_dir() / "checkpoints" / "final_checkpoint.pt"
    print(f"\n{'='*60}")
    print(f"  Pipeline complete!")
    print(f"  Checkpoint: {ckpt}")
    print(f"  Renders: {cfg.scene_output_dir() / 'stage2_renders'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
