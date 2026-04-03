"""Step 6: Stage 1 Optimization — reconstruct visible regions.

Per the paper (Sec 4.3, Stage 1):
  Loss = sum_i L_rec(rendered_i, gt_i)
       + sum_j λ_j * M_α_j * L_rec(rendered_j, repaired_j)
       + sum_j ||A_j ⊙ (1 - M_α_j) ⊙ M_b_j||_1

Stage 1 focuses on visible areas only, using the repair model to generate
pseudo ground truth at M=8 novel views. Novel views are refreshed every 400 iters.

Outputs:
  - outputs/<scene>/stage1_checkpoint.pt   3DGS checkpoint
  - outputs/<scene>/stage1_renders/        rendered novel views
  - outputs/<scene>/stage1_loss.png        loss curve
"""
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RI3DConfig
from gaussian_trainer import (
    GaussianModel, SSIMLoss, LPIPSLoss, depth_correlation_loss,
    reconstruction_loss, PlateauDetector, compute_background_mask,
)
from step4_gaussian_init import generate_elliptical_cameras


def load_repair_pipeline(cfg: RI3DConfig):
    """Load the trained repair model for inference (from shared model dir)."""
    from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        LCMScheduler,
    )
    from peft import PeftModel

    model_dir = cfg.shared_model_dir() / "repair_model"
    dtype = cfg.dtype
    device = cfg.device

    controlnet = ControlNetModel.from_pretrained(cfg.controlnet_model, torch_dtype=dtype)
    controlnet = PeftModel.from_pretrained(controlnet, model_dir)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        cfg.sd_model, controlnet=controlnet, torch_dtype=dtype,
    ).to(device)

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(cfg.lcm_lora)
    pipe.fuse_lora()
    pipe.safety_checker = None

    return pipe


def repair_image(pipe, image_tensor: torch.Tensor, cfg: RI3DConfig) -> torch.Tensor:
    """Run repair model on a rendered image.

    Args:
        image_tensor: (H, W, 3) float tensor in [0, 1]

    Returns:
        repaired: (H, W, 3) float tensor in [0, 1]
    """
    H, W = image_tensor.shape[:2]

    # Convert to PIL at 512x512
    img_np = (image_tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np).resize((512, 512), Image.LANCZOS)

    with torch.no_grad():
        result = pipe(
            prompt="",
            image=img_pil,
            num_inference_steps=cfg.lcm_inference_steps,
            guidance_scale=cfg.lcm_guidance_scale,
        ).images[0]

    # Convert back to tensor at original resolution
    result_np = np.array(result.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
    return torch.from_numpy(result_np).to(image_tensor.device)


def compute_camera_distance_weight(novel_c2w: torch.Tensor,
                                     input_c2ws: torch.Tensor) -> torch.Tensor:
    """Compute λ_j: weight novel cameras by proximity to input cameras.

    Cameras closer to input cameras get higher weight.
    """
    novel_pos = novel_c2w[:3, 3]
    input_pos = input_c2ws[:, :3, 3]  # (N, 3)
    dists = (input_pos - novel_pos.unsqueeze(0)).norm(dim=-1)  # (N,)
    min_dist = dists.min()

    # Exponential decay weight
    weight = torch.exp(-min_dist / (dists.mean() + 1e-8))
    return weight.clamp(0.1, 1.0)


def get_opacity_mask(alpha: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Binary mask from rendered alpha: M_α = (alpha > threshold)."""
    return (alpha.squeeze(-1) > threshold).float()


def get_background_mask(depth: torch.Tensor, cfg: RI3DConfig | None = None) -> torch.Tensor:
    """Background mask via agglomerative clustering on depth (per paper Sec 4.3)."""
    n_clusters = cfg.bg_mask_n_clusters if cfg is not None else 2
    return compute_background_mask(depth, n_clusters)


def run_stage1(cfg: RI3DConfig):
    out_dir = cfg.scene_output_dir()
    render_dir = out_dir / "stage1_renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device

    # Load scene data
    image_paths = torch.load(out_dir / "image_paths.pt", weights_only=False)
    poses = torch.load(out_dir / "dust3r_poses.pt", weights_only=True).float().to(device)
    intrinsics = torch.load(out_dir / "dust3r_intrinsics.pt", weights_only=True).float().to(device)
    gaussians_init = torch.load(out_dir / "init_gaussians.pt", weights_only=True)
    n_images = len(image_paths)

    # Get render resolution
    fused_depth_0 = torch.load(out_dir / "fused_depths" / "fused_depth_000.pt", weights_only=True)
    H, W = fused_depth_0.shape

    # Load GT images
    gt_images = []
    for ip in image_paths:
        img = Image.open(ip).convert("RGB").resize((W, H), Image.LANCZOS)
        gt_images.append(torch.from_numpy(np.array(img)).float().to(device) / 255.0)

    # Load mono depth for depth correlation loss
    mono_depths = []
    for i in range(n_images):
        md = torch.load(out_dir / "mono_depths" / f"mono_depth_{i:03d}.pt", weights_only=True)
        mono_depths.append(md.float().to(device))

    # Initialize model
    model = GaussianModel(gaussians_init, device)
    optimizers = model.setup_optimizers(cfg)
    strategy, strategy_state = model.setup_strategy(cfg, scene_scale=1.0)

    # Loss functions
    ssim_fn = SSIMLoss().to(device)
    lpips_fn = LPIPSLoss(device)

    # Generate novel cameras (use point cloud centroid as scene center)
    scene_center = gaussians_init["means"].mean(dim=0).to(device)
    novel_c2w = generate_elliptical_cameras(poses, cfg.stage1_num_novel_views, scene_center).to(device)
    K_avg = intrinsics.mean(dim=0)

    # Load repair pipeline
    print("Loading repair model for Stage 1...")
    repair_pipe = load_repair_pipeline(cfg)

    # Initialize pseudo ground truth for novel views
    pseudo_gt = [None] * cfg.stage1_num_novel_views
    camera_weights = torch.zeros(cfg.stage1_num_novel_views, device=device)
    for j in range(cfg.stage1_num_novel_views):
        camera_weights[j] = compute_camera_distance_weight(novel_c2w[j], poses)

    # Plateau detector
    plateau = PlateauDetector(cfg.plateau_window, cfg.plateau_threshold, cfg.plateau_min_iters)
    losses_history = []

    print(f"\n=== Stage 1 Optimization ===")
    print(f"Input views: {n_images}, Novel views: {cfg.stage1_num_novel_views}")
    print(f"Max iters: {cfg.stage1_max_iters}, Refresh every: {cfg.stage1_refresh_interval}")

    for step in tqdm(range(cfg.stage1_max_iters), desc="Stage 1"):

        # Refresh pseudo ground truth periodically
        if step % cfg.stage1_refresh_interval == 0:
            print(f"\n  Refreshing pseudo GT at step {step}...")
            with torch.no_grad():
                for j in range(cfg.stage1_num_novel_views):
                    w2c = torch.linalg.inv(novel_c2w[j])
                    r = model.render(w2c, K_avg, H, W)
                    repaired = repair_image(repair_pipe, r["image"], cfg)
                    pseudo_gt[j] = repaired.detach()

            # Save example renders
            if step % (cfg.stage1_refresh_interval * 2) == 0:
                for j in range(min(3, cfg.stage1_num_novel_views)):
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    with torch.no_grad():
                        w2c = torch.linalg.inv(novel_c2w[j])
                        r = model.render(w2c, K_avg, H, W)
                    axes[0].imshow(r["image"].clamp(0,1).cpu().numpy())
                    axes[0].set_title(f"Rendered (step {step})")
                    axes[1].imshow(pseudo_gt[j].cpu().numpy())
                    axes[1].set_title("Pseudo GT (repaired)")
                    axes[2].imshow(r["alpha"].squeeze(-1).cpu().numpy(), cmap="gray")
                    axes[2].set_title("Alpha")
                    for ax in axes: ax.axis("off")
                    fig.savefig(render_dir / f"step{step:05d}_novel{j}.png",
                                dpi=120, bbox_inches="tight")
                    plt.close(fig)

        total_loss = torch.tensor(0.0, device=device)

        # --- Input view loss ---
        ref_idx = step % n_images
        w2c_ref = torch.linalg.inv(poses[ref_idx])
        K_ref = intrinsics[ref_idx]

        result_ref = model.render_for_optim(w2c_ref, K_ref, H, W,
                                             strategy, strategy_state, step)
        ref_loss = reconstruction_loss(result_ref["image"], gt_images[ref_idx],
                                        ssim_fn, lpips_fn, cfg)

        # Depth correlation loss
        d_corr = depth_correlation_loss(result_ref["depth"], mono_depths[ref_idx])
        ref_loss = ref_loss + cfg.loss_depth_corr_weight * d_corr

        total_loss = total_loss + ref_loss

        # --- Novel view loss (if pseudo GT available) ---
        if pseudo_gt[0] is not None:
            nov_idx = step % cfg.stage1_num_novel_views
            w2c_nov = torch.linalg.inv(novel_c2w[nov_idx])

            with torch.no_grad():
                r_nov = model.render(w2c_nov, K_avg, H, W, return_depth=True)

            # Re-render with gradients for the selected novel view
            result_nov = model.render_for_optim(w2c_nov, K_avg, H, W,
                                                 strategy, strategy_state, step)

            # Opacity mask: only enforce loss in visible regions
            alpha_mask = get_opacity_mask(result_nov["alpha"])  # (H, W)
            lambda_j = camera_weights[nov_idx]

            nov_loss = reconstruction_loss(
                result_nov["image"], pseudo_gt[nov_idx],
                ssim_fn, lpips_fn, cfg,
                mask=alpha_mask,
            )
            total_loss = total_loss + lambda_j * nov_loss

            # Opacity regularization: suppress opacity in missing background regions
            bg_mask = get_background_mask(
                result_nov["depth"].squeeze(-1).detach(), cfg
            )
            opacity_reg = (
                result_nov["alpha"].squeeze(-1) * (1 - alpha_mask) * bg_mask
            ).abs().mean()
            total_loss = total_loss + cfg.loss_opacity_reg_weight * opacity_reg

        total_loss.backward()
        model.step_post_backward(step, result_ref["meta"])
        model.optimizer_step()

        loss_val = total_loss.item()
        losses_history.append(loss_val)

        if (step + 1) % 200 == 0:
            print(f"  Step {step+1}: loss = {loss_val:.6f}, "
                  f"n_gaussians = {model.n_gaussians}")

        # Plateau detection
        if plateau.update(loss_val):
            print(f"\n  Plateau detected at step {step+1}, stopping Stage 1.")
            break

    # Save checkpoint
    checkpoint = {
        "gaussians": model.state_dict(),
        "step": step + 1,
        "losses": losses_history,
    }
    ckpt_path = out_dir / "stage1_checkpoint.pt"
    torch.save(checkpoint, ckpt_path)
    print(f"Saved Stage 1 checkpoint to {ckpt_path}")

    # Save loss curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses_history)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Stage 1 Loss")
    fig.savefig(out_dir / "stage1_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Final renders from all input + novel views
    print("Rendering final Stage 1 views...")
    with torch.no_grad():
        for i in range(n_images):
            name = Path(image_paths[i]).stem
            w2c = torch.linalg.inv(poses[i])
            r = model.render(w2c, intrinsics[i], H, W)
            img = r["image"].clamp(0, 1).cpu().numpy()
            plt.imsave(render_dir / f"final_input_{i:03d}_{name}.png", img)

        for j in range(cfg.stage1_num_novel_views):
            w2c = torch.linalg.inv(novel_c2w[j])
            r = model.render(w2c, K_avg, H, W)
            img = r["image"].clamp(0, 1).cpu().numpy()
            alpha = r["alpha"].squeeze(-1).cpu().numpy()
            plt.imsave(render_dir / f"final_novel_{j:03d}.png", img)
            plt.imsave(render_dir / f"final_novel_{j:03d}_alpha.png", alpha, cmap="gray")

    # Cleanup
    del repair_pipe, model
    torch.cuda.empty_cache()

    print(f"\nStage 1 complete! Renders in {render_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 6: Stage 1 optimization")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene image directory")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    cfg = RI3DConfig(scene_dir=Path(args.scene), output_dir=Path(args.output))
    run_stage1(cfg)
