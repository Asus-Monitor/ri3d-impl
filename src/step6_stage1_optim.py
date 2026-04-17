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
    """Load the trained repair model for inference (per-scene ControlNet LoRA).

    Uses img2img ControlNet pipeline. The corrupted render is both the img2img
    input (preserving coarse color/layout through retained signal) and the
    ControlNet condition (providing structural guidance). Strength controls
    how much of the original image survives noise — moderate values destroy
    fine artifacts while preserving global scene consistency.
    """
    from diffusers import (
        StableDiffusionControlNetImg2ImgPipeline,
        ControlNetModel,
        DPMSolverMultistepScheduler,
    )
    from peft import PeftModel

    model_dir = cfg.scene_output_dir() / "repair_model"
    dtype = cfg.dtype
    device = cfg.device

    
    controlnet_lora_dir = model_dir / "controlnet"
    controlnet = ControlNetModel.from_pretrained(cfg.controlnet_model, torch_dtype=dtype, use_safetensors=False)
    controlnet = PeftModel.from_pretrained(controlnet, controlnet_lora_dir)
    controlnet = controlnet.merge_and_unload()

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        cfg.sd_model, controlnet=controlnet, torch_dtype=dtype,
    ).to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True)
    pipe.safety_checker = None

    return pipe


def repair_image(pipe, image_tensor: torch.Tensor, cfg: RI3DConfig,
                 view_index: int = 0,
                 strength_override: float | None = None) -> torch.Tensor:
    """Run repair model on a rendered image.

    Args:
        image_tensor: (H, W, 3) float tensor in [0, 1]
        view_index: deterministic seed offset for this view. Same view_index
            produces same noise → consistent pseudo GT across refresh cycles,
            preventing the optimization from getting conflicting gradients.
        strength_override: if set, overrides cfg.repair_strength. Used by
            Stage 1 for adaptive scheduling (high early → low late).

    Returns:
        repaired: (H, W, 3) float tensor in [0, 1]
    """
    from utils import prepare_for_pipeline as _prepare_for_pipeline

    H, W = image_tensor.shape[:2]

    img_np = (image_tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    img_resized, pipe_h, pipe_w = _prepare_for_pipeline(img_pil)

    
    
    
    
    generator = torch.Generator(device=cfg.device).manual_seed(42 + view_index)

    strength = strength_override if strength_override is not None else cfg.repair_strength

    with torch.no_grad():
        result = pipe(
            prompt=cfg.repair_positive_prompt,
            negative_prompt=cfg.repair_negative_prompt,
            image=img_resized,
            control_image=img_resized,
            strength=strength,
            num_inference_steps=cfg.repair_inference_steps,
            guidance_scale=cfg.repair_guidance_scale,
            controlnet_conditioning_scale=cfg.repair_controlnet_scale,
            generator=generator,
        ).images[0]

    result_np = np.array(result.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
    return torch.from_numpy(result_np).to(image_tensor.device)


def compute_camera_distance_weight(novel_c2w: torch.Tensor,
                                     input_c2ws: torch.Tensor) -> torch.Tensor:
    """Compute λ_j: weight novel cameras by proximity to input cameras.

    Cameras closer to input cameras get higher weight.
    """
    novel_pos = novel_c2w[:3, 3]
    input_pos = input_c2ws[:, :3, 3]  
    dists = (input_pos - novel_pos.unsqueeze(0)).norm(dim=-1)  
    min_dist = dists.min()

    
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

    
    image_paths = cfg.load_image_paths()
    poses = torch.load(out_dir / "dust3r_poses.pt", weights_only=True).float().to(device)
    intrinsics = torch.load(out_dir / "dust3r_intrinsics.pt", weights_only=True).float().to(device)
    gaussians_init = torch.load(out_dir / "init_gaussians.pt", weights_only=True)
    n_images = len(image_paths)

    
    fused_depth_0 = torch.load(out_dir / "fused_depths" / "fused_depth_000.pt", weights_only=True)
    H, W = fused_depth_0.shape

    
    from utils import load_gt_images, load_mono_depths
    gt_images = load_gt_images(image_paths, H, W, device)
    mono_depths = load_mono_depths(out_dir, n_images, device)

    
    model = GaussianModel(gaussians_init, device)
    optimizers = model.setup_optimizers(cfg)

    from utils import compute_scene_scale
    scene_scale = compute_scene_scale(poses)
    print(f"  Scene scale: {scene_scale:.4f}")
    strategy, strategy_state = model.setup_strategy(cfg, scene_scale=scene_scale)

    
    ssim_fn = SSIMLoss().to(device)
    lpips_fn = LPIPSLoss(device)

    
    from step4_gaussian_init import compute_scene_center
    scene_center = compute_scene_center(poses, gaussians_init["means"].to(device))
    
    
    novel_c2w = generate_elliptical_cameras(
        poses, cfg.stage1_num_novel_views, scene_center,
        restrict_to_inputs=True, margin_deg=20.0,
    ).to(device)
    K_avg = intrinsics.mean(dim=0)

    
    input_w2c = torch.linalg.inv(poses)       
    novel_w2c = torch.linalg.inv(novel_c2w)   

    
    print("Loading repair model for Stage 1...")
    repair_pipe = load_repair_pipeline(cfg)

    
    pseudo_gt = [None] * cfg.stage1_num_novel_views
    camera_weights = torch.zeros(cfg.stage1_num_novel_views, device=device)
    for j in range(cfg.stage1_num_novel_views):
        camera_weights[j] = compute_camera_distance_weight(novel_c2w[j], poses)

    
    plateau = PlateauDetector(cfg.plateau_window, cfg.plateau_threshold, cfg.plateau_min_iters)
    losses_history = []

    
    cached_bg_masks = [None] * cfg.stage1_num_novel_views

    
    
    
    s_max = cfg.repair_strength_max  
    s_min = cfg.repair_strength_min  

    print(f"\n=== Stage 1 Optimization ===")
    print(f"Input views: {n_images}, Novel views: {cfg.stage1_num_novel_views}")
    print(f"Max iters: {cfg.stage1_max_iters}, Refresh every: {cfg.stage1_refresh_interval}")
    print(f"Repair strength: {s_max:.2f} → {s_min:.2f} (adaptive)")
    print(f"Densify: {cfg.densify_start}-{cfg.densify_stop}, reset every {cfg.densify_reset_every}")

    for step in tqdm(range(cfg.stage1_max_iters), desc="Stage 1"):

        
        if step % cfg.stage1_refresh_interval == 0:
            
            progress = step / max(cfg.stage1_max_iters - 1, 1)
            cur_strength = s_max + (s_min - s_max) * progress
            print(f"\n  Refreshing pseudo GT at step {step} (strength={cur_strength:.3f})...")
            with torch.no_grad():
                for j in range(cfg.stage1_num_novel_views):
                    r = model.render(novel_w2c[j], K_avg, H, W, return_depth=True)
                    repaired = repair_image(repair_pipe, r["image"], cfg, view_index=j,
                                            strength_override=cur_strength)
                    pseudo_gt[j] = repaired.clamp(0, 1).detach()
                    
                    
                    
                    from utils import estimate_mono_depth
                    mono_d = estimate_mono_depth(repaired, cfg)
                    cached_bg_masks[j] = get_background_mask(
                        mono_d, cfg
                    ).to(device)

                    
                    if j < 3:
                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
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

        
        ref_idx = step % n_images
        w2c_ref = input_w2c[ref_idx]
        K_ref = intrinsics[ref_idx]

        use_lpips = (step % 10 == 0)
        lpips_fn_or_none = lpips_fn if use_lpips else None

        result_ref = model.render_for_optim(w2c_ref, K_ref, H, W,
                                             strategy, strategy_state, step)
        ref_loss = reconstruction_loss(result_ref["image"], gt_images[ref_idx],
                                        ssim_fn, lpips_fn_or_none, cfg)

        
        d_corr = depth_correlation_loss(result_ref["depth"], mono_depths[ref_idx])
        ref_loss = ref_loss + cfg.loss_depth_corr_weight * d_corr

        
        ref_loss.backward()
        model.step_post_backward(step, result_ref["meta"])
        loss_val = ref_loss.item()

        
        
        
        if pseudo_gt[0] is not None:
            for nov_idx in range(cfg.stage1_num_novel_views):
                w2c_nov = novel_w2c[nov_idx]

                result_nov = model.render_for_loss(w2c_nov, K_avg, H, W, render_mode="RGB")

                
                alpha_mask = get_opacity_mask(result_nov["alpha"])  
                lambda_j = camera_weights[nov_idx]

                
                
                
                nov_loss = reconstruction_loss(
                    result_nov["image"], pseudo_gt[nov_idx],
                    ssim_fn, lpips_fn_or_none, cfg,
                    mask=alpha_mask,
                )
                view_loss = (lambda_j / n_images) * nov_loss

                
                bg_mask = cached_bg_masks[nov_idx]
                if bg_mask is not None:
                    opacity_reg = (
                        result_nov["alpha"].squeeze(-1) * (1 - alpha_mask) * bg_mask
                    ).abs().mean()
                    view_loss = view_loss + (cfg.loss_opacity_reg_weight / n_images) * opacity_reg

                view_loss.backward()
                loss_val += view_loss.item()

        
        model.optimizer_step()
        losses_history.append(loss_val)

        if (step + 1) % 200 == 0:
            print(f"  Step {step+1}: loss = {loss_val:.6f}, "
                  f"n_gaussians = {model.n_gaussians}")

        
        if plateau.update(loss_val):
            print(f"\n  Plateau detected at step {step+1}, stopping Stage 1.")
            break

    
    checkpoint = {
        "gaussians": model.state_dict(),
        "step": step + 1,
        "losses": losses_history,
    }
    ckpt_path = out_dir / "stage1_checkpoint.pt"
    torch.save(checkpoint, ckpt_path)
    print(f"Saved Stage 1 checkpoint to {ckpt_path}")

    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses_history)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Stage 1 Loss")
    fig.savefig(out_dir / "stage1_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    
    print("Rendering final Stage 1 views...")
    with torch.no_grad():
        for i in range(n_images):
            name = Path(image_paths[i]).stem
            r = model.render(input_w2c[i], intrinsics[i], H, W)
            img = r["image"].clamp(0, 1).cpu().numpy()
            plt.imsave(render_dir / f"final_input_{i:03d}_{name}.png", img)

        for j in range(cfg.stage1_num_novel_views):
            r = model.render(novel_w2c[j], K_avg, H, W)
            img = r["image"].clamp(0, 1).cpu().numpy()
            alpha = r["alpha"].squeeze(-1).cpu().numpy()
            plt.imsave(render_dir / f"final_novel_{j:03d}.png", img)
            plt.imsave(render_dir / f"final_novel_{j:03d}_alpha.png", alpha, cmap="gray")

    
    del repair_pipe, model
    torch.cuda.empty_cache()

    print(f"\nStage 1 complete! Renders in {render_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 6: Stage 1 optimization")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene image directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    cfg = RI3DConfig(scene_dir=Path(args.scene),
                     output_dir=Path(args.output) if args.output else None)
    run_stage1(cfg)
