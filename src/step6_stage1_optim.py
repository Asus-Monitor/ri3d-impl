"""Step 6: Stage 1 Optimization — reconstruct visible regions.

Per the paper (Sec 4.3, Eq. 3):
  L = sum_i L_rec(rendered_i, gt_i)
    + sum_j λ_j * M_α_j * L_rec(rendered_j, repaired_j)
    + sum_j ||A_j ⊙ (1 - M_α_j) ⊙ M_b_j||_1

Stage 1 uses the repair model to generate pseudo ground truth at M=8 novel
views. Repaired views are refreshed every 400 iters (paper §8.3).

Anti-spiral mechanism (critical for avoiding "AI hallucination" drift across
refreshes):
  • M_α is LIVE (recomputed each step from current render's alpha channel).
    It must be free to shrink under opacity reg pressure.
  • M_b is cached at refresh (from mono-depth of the repaired image).
  • Opacity reg term (A · (1-M_α) · M_b) runs at full weight. It's the
    anti-ratchet: without it, repair hallucinations in background regions
    grow alpha → expand M_α → get supervised → compound.

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
    """Load the trained repair model for inference (per-scene fine-tuned ControlNet).

    Uses img2img ControlNet pipeline. The corrupted render is both the img2img
    input (preserving coarse color/layout through retained signal) and the
    ControlNet condition (providing structural guidance). Strength controls
    how much of the original image survives noise — moderate values destroy
    fine artifacts while preserving global scene consistency.
    """
    from diffusers import (
        StableDiffusionControlNetImg2ImgPipeline,
        ControlNetModel,
        DDIMScheduler,
    )

    model_dir = cfg.scene_output_dir() / "repair_model"
    dtype = cfg.dtype
    device = cfg.device

    controlnet = ControlNetModel.from_pretrained(
        model_dir / "controlnet", torch_dtype=dtype)

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        cfg.sd_model, controlnet=controlnet, torch_dtype=dtype,
    ).to(device)

    # GaussianObject uses DDIM with eta=1.0 (stochastic). The eta is passed
    # through the pipe call below, not on the scheduler config.
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None

    try:
        import xformers  # noqa: F401
        pipe.enable_xformers_memory_efficient_attention()
    except (ImportError, ModuleNotFoundError, AttributeError, Exception):
        pass

    return pipe


def repair_image(pipe, image_tensor: torch.Tensor, cfg: RI3DConfig,
                 view_index: int = 0,
                 strength_override: float | None = None,
                 eta: float | None = None) -> torch.Tensor:
    """Run repair model on a rendered image.

    Args:
        image_tensor: (H, W, 3) float tensor in [0, 1]
        view_index: kept for caller compatibility but no longer affects the
            DDIM seed. We use a constant seed across all views and refreshes:
            with eta=1.0, same noise sequence + view-dependent inputs → the
            model's output differs across views only via input differences,
            not via independently-sampled posterior modes. Per-view seed
            offsets caused inter-view inconsistency once the blob-augmented
            repair model started doing real corrections (its output then
            depended on the noise draw, not just the input). Constant seed
            still gives temporal stability per view: across refreshes, the
            input evolves with 3DGS, output evolves with input — noise itself
            doesn't drift the output.
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

    # Constant seed across views and refreshes (see view_index docstring).
    del view_index  # acknowledged, not used for seeding
    generator = torch.Generator(device=cfg.device).manual_seed(42)

    strength = strength_override if strength_override is not None else cfg.repair_strength
    eta_val = eta if eta is not None else cfg.repair_eta_optim

    # CFG=1.0 (GaussianObject convention) — negative prompt would be ignored,
    # so we don't pass it; saves a text-encoder forward.
    with torch.no_grad():
        result = pipe(
            prompt=cfg.repair_positive_prompt,
            image=img_resized,
            control_image=img_resized,
            strength=strength,
            num_inference_steps=cfg.repair_inference_steps,
            guidance_scale=cfg.repair_guidance_scale,
            controlnet_conditioning_scale=cfg.repair_controlnet_scale,
            eta=eta_val,
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

    # M_b (background mask) is cached per-refresh: paper §4.3 says it's
    # computed from agglomerative clustering on mono-depth of the *repaired*
    # image, which only updates at refresh time. M_α is NOT cached — it's
    # recomputed live each step from the current render's alpha channel
    # (paper Eq. 3, second term). Live M_α is load-bearing: as opacity reg
    # shrinks alpha in background regions, M_α shrinks too, which in turn
    # reduces the supervised area → prevents the content ratchet.
    cached_bg_masks = [None] * cfg.stage1_num_novel_views

    s_max = cfg.stage1_repair_strength_max
    s_min = cfg.stage1_repair_strength_min
    n_nov = cfg.stage1_num_novel_views
    warmup = cfg.stage1_warmup_iters

    print(f"\n=== Stage 1 Optimization ===")
    print(f"Input views: {n_images}, Novel views: {n_nov}")
    print(f"Max iters: {cfg.stage1_max_iters}, warmup: {warmup}, refresh every {cfg.stage1_refresh_interval} iters after warmup")
    print(f"Repair strength: {s_max:.2f} → {s_min:.2f} (adaptive per refresh)")
    print(f"Densify: {cfg.densify_start}-{cfg.densify_stop}, reset every {cfg.densify_reset_every}")

    for step in tqdm(range(cfg.stage1_max_iters), desc="Stage 1"):

        # Refresh pseudo-GT on a shifted schedule: no refreshes during warmup,
        # first refresh at step == warmup, then every stage1_refresh_interval.
        # At each refresh: render all M novel views, repair each, blend into
        # pseudo-GT via EMA; recompute M_b from mono-depth of the repaired image.
        # Strength decays across iterations (big corrections early, small late).
        #
        # Warmup rationale: repair is fine-tuned on renders from mid-training
        # 3DGS (§4.2 uses iter 6000-10000 leave-one-out renders). A fresh-init
        # per-pixel 3DGS rendered at a novel view is OOD for that model, and
        # OOD inputs to img2img ControlNet → SD prior dominates → hallucinated
        # pseudo-GT that anchors everything downstream. 1000 iters of input-only
        # optimization gets the render into-distribution before first repair.
        #
        # EMA blend (pseudo_gt_new = (1-α) * pseudo_gt_old + α * repair(render))
        # is our defense against Loop A drift: SD1.5 isn't identity-on-clean,
        # so iterated repair(·) drifts toward SD's fixed point. Blending caps
        # each refresh's influence at 0.3 of the delta, bounding cross-refresh
        # drift. First refresh uses full repair (nothing to blend with yet).
        in_refresh_phase = step >= warmup
        steps_since_warmup = step - warmup
        is_refresh_step = (in_refresh_phase
                            and steps_since_warmup % cfg.stage1_refresh_interval == 0)
        if is_refresh_step:
            progress = step / max(cfg.stage1_max_iters - 1, 1)
            cur_strength = s_max + (s_min - s_max) * progress
            ema_alpha = 1.0 if step == warmup else 0.3
            print(f"\n  Refresh at step {step} (strength={cur_strength:.3f}, ema_alpha={ema_alpha:.2f})...")
            coverage_sum = 0.0
            with torch.no_grad():
                for j in range(n_nov):
                    r = model.render(novel_w2c[j], K_avg, H, W, return_depth=True)
                    repaired = repair_image(repair_pipe, r["image"], cfg, view_index=j,
                                            strength_override=cur_strength)
                    repaired = repaired.clamp(0, 1)
                    if pseudo_gt[j] is None:
                        pseudo_gt[j] = repaired.detach()
                    else:
                        pseudo_gt[j] = ((1 - ema_alpha) * pseudo_gt[j]
                                        + ema_alpha * repaired).detach()

                    from utils import estimate_mono_depth
                    mono_d = estimate_mono_depth(repaired, cfg)
                    cached_bg_masks[j] = get_background_mask(mono_d, cfg).to(device)

                    # Diagnostic: track M_α coverage across refreshes. If this
                    # grows monotonically, the content ratchet is winning →
                    # bump loss_opacity_reg_weight in config.
                    coverage_sum += get_opacity_mask(r["alpha"]).mean().item()

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
            print(f"    M_α coverage (mean over {n_nov} novel views): {coverage_sum/n_nov:.4f}")


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

        # Novel-view terms (Eq. 3, terms 2 & 3): all M views, per-paper weights.
        # No /N or /n_nov scaling — each view contributes λ_j of signal, and
        # the opacity reg runs at full weight to counteract M_α inflation.
        if pseudo_gt[0] is not None:
            for nov_idx in range(n_nov):
                w2c_nov = novel_w2c[nov_idx]

                result_nov = model.render_for_loss(w2c_nov, K_avg, H, W, render_mode="RGB")

                # Live M_α from CURRENT render — not cached. As opacity reg
                # shrinks alpha in background regions, M_α follows, which
                # removes repair supervision from hallucination-prone regions.
                alpha_mask = get_opacity_mask(result_nov["alpha"]).detach()
                lambda_j = camera_weights[nov_idx]

                nov_loss = reconstruction_loss(
                    result_nov["image"], pseudo_gt[nov_idx],
                    ssim_fn, lpips_fn_or_none, cfg,
                    mask=alpha_mask,
                )
                view_loss = lambda_j * nov_loss

                bg_mask = cached_bg_masks[nov_idx]
                if bg_mask is not None:
                    # Anti-ratchet: A · (1-M_α) · M_b pushed toward zero.
                    # Must be strong enough to shrink M_α faster than repair
                    # supervision inflates it.
                    opacity_reg = (
                        result_nov["alpha"].squeeze(-1) * (1 - alpha_mask) * bg_mask
                    ).abs().mean()
                    view_loss = view_loss + cfg.loss_opacity_reg_weight * opacity_reg

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
