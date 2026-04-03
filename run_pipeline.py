"""RI3D Full Pipeline Runner.

Usage:
  python run_pipeline.py --scene dataset/garden
  python run_pipeline.py --scene dataset/garden --step 3   # resume from step 3
  python run_pipeline.py --scene dataset/garden --step 1 --only  # run only step 1
"""
import argparse
from pathlib import Path

from config import RI3DConfig


STEPS = {
    1: ("DUSt3R: camera poses + coarse depth", "step1_dust3r", "run_dust3r"),
    2: ("Depth Anything V2: monocular depth", "step2_mono_depth", "run_mono_depth"),
    3: ("Poisson depth fusion", "step3_depth_fusion", "run_depth_fusion"),
    4: ("Gaussian initialization + render", "step4_gaussian_init", "run_step4"),
    5: ("Repair model training", "step5_repair_model", "run_step5"),
    6: ("Stage 1 optimization", "step6_stage1_optim", "run_stage1"),
    7: ("Inpainting model training", "step7_inpainting_model", "run_step7"),
    8: ("Stage 2 optimization", "step8_stage2_optim", "run_stage2"),
}


def run_step(step_num: int, cfg: RI3DConfig):
    name, module_name, func_name = STEPS[step_num]
    print(f"\n{'='*60}")
    print(f"  Step {step_num}: {name}")
    print(f"{'='*60}\n")

    module = __import__(module_name)
    func = getattr(module, func_name)
    func(cfg)


def main():
    parser = argparse.ArgumentParser(description="RI3D Pipeline")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene image directory")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--step", type=int, default=1, help="Start from this step (1-8)")
    parser.add_argument("--only", action="store_true", help="Run only the specified step")
    args = parser.parse_args()

    cfg = RI3DConfig(scene_dir=Path(args.scene), output_dir=Path(args.output))

    # Ensure output dirs exist
    cfg.scene_output_dir()
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"RI3D Pipeline")
    print(f"  Scene: {cfg.scene_dir}")
    print(f"  Output: {cfg.output_dir}")
    print(f"  Device: {cfg.device}, dtype: {cfg.dtype}")

    if args.only:
        run_step(args.step, cfg)
    else:
        for s in range(args.step, 9):
            run_step(s, cfg)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete!")
    print(f"  Final checkpoint: {cfg.scene_output_dir() / 'checkpoints' / 'final_checkpoint.pt'}")
    print(f"  Final renders: {cfg.scene_output_dir() / 'stage2_renders'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
