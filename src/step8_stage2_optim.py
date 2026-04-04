"""Step 8: Stage 2 Optimization — inpaint missing regions and finalize.

Per the paper (Sec 4.3, Stage 2):
  1. Select K non-overlapping novel views with missing regions
  2. Inpaint missing areas using personalized inpainting model
  3. Project inpainted content into 3D using monocular depth + alignment
  4. Continue optimization with repair model on all novel views
  5. Repeat inpaint-optimize cycle until missing areas are filled

Loss_stage2 = sum_i L_rec(rendered_i, gt_i)
            + sum_j λ_j L_rec(rendered_j, repaired_j)      [no visibility mask]
            + sum_k (1-M_α_k) ⊙ M_b_k ⊙ Lp(rendered_k, inpainted_k)

Outputs:
  - outputs/<scene>/stage2_checkpoint.pt   final 3DGS checkpoint
  - outputs/<scene>/stage2_renders/        final renders
  - outputs/<scene>/stage2_loss.png        loss curve
"""
import argparse
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
from step4_gaussian_init import generate_elliptical_cameras, unproject_depth
from step6_stage1_optim import (
    load_repair_pipeline, repair_image,
    compute_camera_distance_weight, get_opacity_mask,
)


def load_inpainting_pipeline(cfg: RI3DConfig):
    """Load the trained inpainting model for inference (from shared model dir)."""
    from diffusers import StableDiffusionInpaintPipeline, LCMScheduler
    from peft import PeftModel

    model_dir = cfg.shared_model_dir() / "inpainting_model"
    dtype = cfg.dtype
    device = cfg.device

    inpaint_model_id = "runwayml/stable-diffusion-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        inpaint_model_id, torch_dtype=dtype
    ).to(device)

    pipe.unet = PeftModel.from_pretrained(pipe.unet, model_dir)
    pipe.unet = pipe.unet.merge_and_unload()
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(cfg.lcm_lora, adapter_name="lcm")
    pipe.safety_checker = None

    return pipe


def inpaint_missing_regions(pipe, rendered_image: torch.Tensor,
                             alpha_mask: torch.Tensor, bg_mask: torch.Tensor,
                             cfg: RI3DConfig) -> torch.Tensor:
    """Inpaint missing regions in a rendered image.

    Args:
        rendered_image: (H, W, 3) rendered image
        alpha_mask: (H, W) binary mask, 1 = visible (has Gaussians)
        bg_mask: (H, W) binary mask, 1 = background region

    Returns:
        inpainted: (H, W, 3) image with missing regions filled
    """
    H, W = rendered_image.shape[:2]
    missing = ((1 - alpha_mask) * bg_mask)

    if missing.sum() < 100:
        return rendered_image.clone()

    img_np = (rendered_image.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    mask_np = (missing.cpu().numpy() * 255).astype(np.uint8)

    img_pil = Image.fromarray(img_np).resize((512, 512), Image.LANCZOS)
    mask_pil = Image.fromarray(mask_np, mode="L").resize((512, 512), Image.NEAREST)

    with torch.no_grad():
        result = pipe(
            prompt="",
            image=img_pil,
            mask_image=mask_pil,
            num_inference_steps=cfg.lcm_inference_steps,
            guidance_scale=cfg.lcm_guidance_scale,
        ).images[0]

    result_np = np.array(result.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
    inpainted = torch.from_numpy(result_np).to(rendered_image.device)

    # Composite: keep visible regions from render, use inpainted for missing
    alpha_3ch = alpha_mask.unsqueeze(-1)
    composite = rendered_image * alpha_3ch + inpainted * (1 - alpha_3ch)

    return composite


def estimate_mono_depth(image_tensor: torch.Tensor, cfg: RI3DConfig) -> torch.Tensor:
    """Run monocular depth estimation on an image tensor.

    Args:
        image_tensor: (H, W, 3) float tensor in [0, 1]
    Returns:
        depth: (H, W) float tensor
    """
    from transformers import pipeline as hf_pipeline

    H, W = image_tensor.shape[:2]
    img_np = (image_tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)

    # Use cached pipeline
    if not hasattr(estimate_mono_depth, "_pipe"):
        estimate_mono_depth._pipe = hf_pipeline(
            "depth-estimation", model=cfg.depth_model,
            device=cfg.device, torch_dtype=cfg.dtype,
        )
    with torch.no_grad():
        result = estimate_mono_depth._pipe(img_pil)

    mono = result["predicted_depth"]
    if mono.dim() == 3:
        mono = mono.squeeze(0)
    mono = F.interpolate(
        mono.unsqueeze(0).unsqueeze(0).float(),
        size=(H, W), mode="bilinear", align_corners=False,
    ).squeeze()
    return mono


def clear_mono_depth_cache():
    if hasattr(estimate_mono_depth, "_pipe"):
        del estimate_mono_depth._pipe
        torch.cuda.empty_cache()


def project_inpainted_to_3d(inpainted_image: torch.Tensor, missing_mask: torch.Tensor,
                              rendered_depth: torch.Tensor, c2w: torch.Tensor,
                              K: torch.Tensor, cfg: RI3DConfig,
                              max_new_gaussians: int = 3000) -> dict | None:
    """Project inpainted pixels into 3D as new Gaussians.

    Uses monocular depth aligned to rendered depth via simple linear fit
    (fast alternative to full Poisson fusion for the refresh cycle).
    Subsamples to max_new_gaussians to prevent unbounded growth.
    """
    device = inpainted_image.device
    H, W = inpainted_image.shape[:2]

    # Skip if very few missing pixels
    n_missing = (missing_mask > 0.5).sum().item()
    if n_missing < 100:
        return None

    # Monocular depth on inpainted image
    mono_depth = estimate_mono_depth(inpainted_image, cfg)  # (H, W) on GPU

    # Align mono depth to rendered depth scale using visible (non-missing) pixels
    visible = (missing_mask < 0.5)
    rd = rendered_depth.squeeze()

    # Simple linear alignment: mono_aligned = scale * mono + shift
    # Fitted only on visible pixels where rendered depth is valid
    valid = visible & (rd > 0.01)
    if valid.sum() < 100:
        return None

    mono_valid = mono_depth[valid].float()
    rd_valid = rd[valid].float()

    # Least-squares: [mono, 1] @ [scale, shift] = rd
    A = torch.stack([mono_valid, torch.ones_like(mono_valid)], dim=1)
    result = torch.linalg.lstsq(A, rd_valid)
    scale, shift = result.solution[0].item(), result.solution[1].item()

    aligned_depth = (mono_depth.float() * scale + shift).clamp(min=0.01)

    # Composite depth: use rendered depth where visible, aligned mono where missing
    fused = torch.where(visible, rd, aligned_depth).to(device)

    # Get missing pixel indices
    mask_flat = missing_mask.reshape(-1) > 0.5
    n_total = mask_flat.sum().item()
    if n_total == 0:
        return None

    # Subsample if too many missing pixels
    if n_total > max_new_gaussians:
        indices = torch.where(mask_flat)[0]
        perm = torch.randperm(n_total, device=device)[:max_new_gaussians]
        subsample_mask = torch.zeros_like(mask_flat)
        subsample_mask[indices[perm]] = True
        mask_flat = subsample_mask

    # Unproject on GPU
    pts = unproject_depth(fused, K, c2w)
    colors = inpainted_image.reshape(-1, 3)

    pts = pts[mask_flat]
    colors = colors[mask_flat]
    depths_valid = fused.reshape(-1)[mask_flat]

    # Scale from depth and focal length
    fx = K[0, 0].item()
    pixel_sizes = depths_valid / fx * cfg.gaussian_scale_factor
    scales = pixel_sizes.unsqueeze(-1).expand(-1, 3)

    n = pts.shape[0]
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0
    opacities = torch.full((n,), 0.5, device=device)

    return {
        "means": pts,
        "scales": torch.log(scales.clamp(min=1e-8)),
        "quats": quats,
        "opacities": torch.logit(opacities.clamp(1e-4, 1 - 1e-4)),
        "colors": colors,
    }


def add_gaussians_to_model(model: GaussianModel, new_gaussians: dict):
    """Append new Gaussians to existing model."""
    device = model.device
    for key in model.params:
        old = model.params[key].data
        new = new_gaussians[key].float().to(device)
        combined = torch.cat([old, new], dim=0)
        model.params[key] = torch.nn.Parameter(combined)
    return model


def select_inpaint_views(renders_cache: dict, n_views: int, k_inpaint: int,
                          last_inpainted: set) -> list[int]:
    """Select K views with largest missing regions, rotating to avoid always picking same views.

    Per paper: "different subset of K images" each cycle, non-overlapping content.
    """
    # Score each view by missing area, penalize recently inpainted views
    scores = {}
    for j in range(n_views):
        rc = renders_cache[j]
        alpha_mask = rc["alpha_mask"]
        bg_mask = rc["bg_mask"]
        missing = ((1 - alpha_mask) * bg_mask).sum().item()
        # Penalize recently inpainted views to encourage rotation
        if j in last_inpainted:
            missing *= 0.3
        scores[j] = missing

    # Pick top K by missing area
    ranked = sorted(scores.keys(), key=lambda j: scores[j], reverse=True)
    selected = ranked[:k_inpaint]

    # Only inpaint views that actually have significant missing regions
    selected = [j for j in selected if scores[j] > 100]
    return selected


def run_stage2(cfg: RI3DConfig):
    out_dir = cfg.scene_output_dir()
    render_dir = out_dir / "stage2_renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device

    # Load scene data
    image_paths = torch.load(out_dir / "image_paths.pt", weights_only=False)
    poses = torch.load(out_dir / "dust3r_poses.pt", weights_only=True).float().to(device)
    intrinsics = torch.load(out_dir / "dust3r_intrinsics.pt", weights_only=True).float().to(device)
    n_images = len(image_paths)

    # Load Stage 1 checkpoint
    stage1_ckpt = torch.load(out_dir / "stage1_checkpoint.pt", weights_only=True)
    fused_depth_0 = torch.load(out_dir / "fused_depths" / "fused_depth_000.pt", weights_only=True)
    H, W = fused_depth_0.shape

    # Load GT images
    gt_images = []
    for ip in image_paths:
        img = Image.open(ip).convert("RGB").resize((W, H), Image.LANCZOS)
        gt_images.append(torch.from_numpy(np.array(img)).float().to(device) / 255.0)

    # Load mono depths for reference
    mono_depths = []
    for i in range(n_images):
        md = torch.load(out_dir / "mono_depths" / f"mono_depth_{i:03d}.pt", weights_only=True)
        mono_depths.append(md.float().to(device))

    # Initialize model from Stage 1
    model = GaussianModel(stage1_ckpt["gaussians"], device)
    optimizers = model.setup_optimizers(cfg)

    # Compute scene scale from camera positions
    cam_positions = poses[:, :3, 3]
    scene_scale = (cam_positions - cam_positions.mean(dim=0)).norm(dim=1).mean().item()
    scene_scale = max(scene_scale, 0.1)
    print(f"  Scene scale: {scene_scale:.4f}")
    strategy, strategy_state = model.setup_strategy(cfg, scene_scale=scene_scale)

    # Loss functions
    ssim_fn = SSIMLoss().to(device)
    lpips_fn = LPIPSLoss(device)

    # Generate novel cameras
    from step4_gaussian_init import compute_scene_center
    scene_center = compute_scene_center(poses, stage1_ckpt["gaussians"]["means"].to(device))
    novel_c2w = generate_elliptical_cameras(poses, cfg.stage2_num_novel_views, scene_center).to(device)
    K_avg = intrinsics.mean(dim=0)

    camera_weights = torch.zeros(cfg.stage2_num_novel_views, device=device)
    for j in range(cfg.stage2_num_novel_views):
        camera_weights[j] = compute_camera_distance_weight(novel_c2w[j], poses)

    # Precompute w2c matrices
    input_w2c = torch.linalg.inv(poses)       # (N, 4, 4)
    novel_w2c = torch.linalg.inv(novel_c2w)   # (M, 4, 4)

    # Diffusion pipelines — loaded lazily, freed when no longer needed
    inpaint_pipe = None
    repair_pipe = None

    # Storage
    pseudo_gt = [None] * cfg.stage2_num_novel_views
    inpainted_images = [None] * cfg.stage2_num_novel_views
    cached_bg_masks = [None] * cfg.stage2_num_novel_views
    last_inpainted = set()  # track which views were inpainted last cycle

    plateau = PlateauDetector(cfg.plateau_window, cfg.plateau_threshold, cfg.plateau_min_iters)
    losses_history = []

    print(f"\n=== Stage 2 Optimization ===")
    print(f"Max iters: {cfg.stage2_max_iters}")
    print(f"Novel views: {cfg.stage2_num_novel_views}, Inpaint views: {cfg.stage2_num_inpaint_views}")
    print(f"Inpaint interval: {cfg.stage2_inpaint_interval}, cutoff: {cfg.stage2_inpaint_cutoff}")

    for step in tqdm(range(cfg.stage2_max_iters), desc="Stage 2"):

        # === Refresh cycle ===
        if step % cfg.stage2_inpaint_interval == 0:
            print(f"\n  Refresh cycle at step {step}...")

            with torch.no_grad():
                # Phase 1: Render all novel views (fast — just gsplat)
                renders_cache = {}
                for j in range(cfg.stage2_num_novel_views):
                    r = model.render(novel_w2c[j], K_avg, H, W, return_depth=True)
                    alpha_mask = get_opacity_mask(r["alpha"])
                    bg_mask = compute_background_mask(
                        r["depth"].squeeze(-1).detach(), cfg.bg_mask_n_clusters
                    )
                    cached_bg_masks[j] = bg_mask.to(device)
                    renders_cache[j] = {
                        "image": r["image"].detach(),
                        "depth": r["depth"].detach(),
                        "alpha": r["alpha"].detach(),
                        "alpha_mask": alpha_mask,
                        "bg_mask": bg_mask,
                    }

                # Phase 2: Inpaint selected views (before cutoff)
                do_inpaint = step < cfg.stage2_inpaint_cutoff
                if do_inpaint:
                    # Select K views with most missing content, rotating subsets
                    inpaint_views = select_inpaint_views(
                        renders_cache, cfg.stage2_num_novel_views,
                        cfg.stage2_num_inpaint_views, last_inpainted,
                    )
                    last_inpainted = set(inpaint_views)

                    if inpaint_views:
                        # Load inpaint pipeline on first use
                        if inpaint_pipe is None:
                            print("  Loading inpainting pipeline...")
                            inpaint_pipe = load_inpainting_pipeline(cfg)

                        for j in inpaint_views:
                            rc = renders_cache[j]
                            inpainted = inpaint_missing_regions(
                                inpaint_pipe, rc["image"], rc["alpha_mask"],
                                rc["bg_mask"], cfg,
                            )
                            inpainted_images[j] = inpainted.detach()

                            # Project into 3D (subsampled, fast linear alignment)
                            missing_mask = ((1 - rc["alpha_mask"]) * rc["bg_mask"]).detach()
                            new_gs = project_inpainted_to_3d(
                                inpainted, missing_mask,
                                rc["depth"], novel_c2w[j], K_avg, cfg,
                            )
                            if new_gs is not None:
                                n_before = model.n_gaussians
                                add_gaussians_to_model(model, new_gs)
                                optimizers = model.setup_optimizers(cfg)
                                strategy, strategy_state = model.setup_strategy(
                                    cfg, scene_scale=scene_scale,
                                )
                                print(f"    View {j}: +{model.n_gaussians - n_before} "
                                      f"Gaussians (total: {model.n_gaussians})")
                else:
                    # Past cutoff — free inpaint + depth models
                    if inpaint_pipe is not None:
                        del inpaint_pipe
                        inpaint_pipe = None
                        clear_mono_depth_cache()
                        torch.cuda.empty_cache()
                        print("  Freed inpainting + depth models (past cutoff)")

                # Phase 3: Repair all novel views
                if repair_pipe is None:
                    print("  Loading repair pipeline...")
                    repair_pipe = load_repair_pipeline(cfg)

                for j in range(cfg.stage2_num_novel_views):
                    rc = renders_cache[j]
                    # Repair inpainted image if available, otherwise raw render
                    if inpainted_images[j] is not None:
                        repaired = repair_image(repair_pipe, inpainted_images[j], cfg)
                    else:
                        repaired = repair_image(repair_pipe, rc["image"], cfg)
                    pseudo_gt[j] = repaired.detach()

                del renders_cache

            # Save example renders (not every cycle — every 4th)
            if step % (cfg.stage2_inpaint_interval * 4) == 0:
                for j in range(min(3, cfg.stage2_num_novel_views)):
                    with torch.no_grad():
                        r = model.render(novel_w2c[j], K_avg, H, W)

                    n_cols = 4 if inpainted_images[j] is not None else 3
                    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
                    axes[0].imshow(r["image"].clamp(0,1).cpu().numpy())
                    axes[0].set_title(f"Rendered (step {step})")
                    axes[1].imshow(pseudo_gt[j].cpu().numpy())
                    axes[1].set_title("Pseudo GT")
                    axes[2].imshow(r["alpha"].squeeze(-1).cpu().numpy(), cmap="gray")
                    axes[2].set_title("Alpha")
                    if inpainted_images[j] is not None:
                        axes[3].imshow(inpainted_images[j].cpu().numpy())
                        axes[3].set_title("Inpainted")
                    for ax in axes: ax.axis("off")
                    fig.savefig(render_dir / f"step{step:05d}_novel{j}.png",
                                dpi=120, bbox_inches="tight")
                    plt.close(fig)

        total_loss = torch.tensor(0.0, device=device)

        # --- Input view loss ---
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
        total_loss = total_loss + ref_loss

        # --- Novel view loss (no visibility mask in Stage 2) ---
        if pseudo_gt[0] is not None:
            nov_idx = step % cfg.stage2_num_novel_views
            w2c_nov = novel_w2c[nov_idx]

            result_nov = model.render_for_loss(w2c_nov, K_avg, H, W)
            lambda_j = camera_weights[nov_idx]

            nov_loss = reconstruction_loss(
                result_nov["image"], pseudo_gt[nov_idx],
                ssim_fn, lpips_fn_or_none, cfg,
            )
            total_loss = total_loss + lambda_j * nov_loss

            # Inpainting consistency loss: (1-M_α) ⊙ M_b ⊙ Lp (per paper)
            # Use CURRENT alpha mask, not stale cached one — as optimization
            # fills regions, the mask should shrink automatically
            if inpainted_images[nov_idx] is not None and cached_bg_masks[nov_idx] is not None:
                alpha_mask_now = get_opacity_mask(result_nov["alpha"])
                mask_ip = (1 - alpha_mask_now) * cached_bg_masks[nov_idx]
                n_masked = mask_ip.sum().clamp(min=1)
                mask_3ch = mask_ip.unsqueeze(-1)
                inpaint_loss = (
                    (result_nov["image"] - inpainted_images[nov_idx]).abs() * mask_3ch
                ).sum() / (n_masked * 3)
                total_loss = total_loss + 0.5 * inpaint_loss

        total_loss.backward()
        model.step_post_backward(step, result_ref["meta"])
        model.optimizer_step()

        loss_val = total_loss.item()
        losses_history.append(loss_val)

        if (step + 1) % 200 == 0:
            print(f"  Step {step+1}: loss = {loss_val:.6f}, "
                  f"n_gaussians = {model.n_gaussians}")

        if plateau.update(loss_val):
            print(f"\n  Plateau detected at step {step+1}, stopping Stage 2.")
            break

    # Save final checkpoint
    checkpoint = {
        "gaussians": model.state_dict(),
        "step": step + 1,
        "losses": losses_history,
        "novel_cameras": novel_c2w.cpu(),
        "intrinsics": intrinsics.cpu(),
        "poses": poses.cpu(),
    }
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "final_checkpoint.pt"
    torch.save(checkpoint, ckpt_path)
    print(f"Saved final checkpoint to {ckpt_path}")
    torch.save(checkpoint, out_dir / "stage2_checkpoint.pt")

    # Loss curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses_history)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Stage 2 Loss")
    fig.savefig(out_dir / "stage2_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Final renders
    print("Rendering final views...")
    with torch.no_grad():
        for i in range(n_images):
            name = Path(image_paths[i]).stem
            r = model.render(input_w2c[i], intrinsics[i], H, W)
            img = r["image"].clamp(0, 1).cpu().numpy()
            plt.imsave(render_dir / f"final_input_{i:03d}_{name}.png", img)

        for j in range(cfg.stage2_num_novel_views):
            r = model.render(novel_w2c[j], K_avg, H, W)
            img = r["image"].clamp(0, 1).cpu().numpy()
            alpha = r["alpha"].squeeze(-1).cpu().numpy()
            plt.imsave(render_dir / f"final_novel_{j:03d}.png", img)
            plt.imsave(render_dir / f"final_novel_{j:03d}_alpha.png", alpha, cmap="gray")

    # Cleanup
    clear_mono_depth_cache()
    if repair_pipe is not None:
        del repair_pipe
    if inpaint_pipe is not None:
        del inpaint_pipe
    del model
    torch.cuda.empty_cache()

    print(f"\nStage 2 complete! Final renders in {render_dir}")
    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 8: Stage 2 optimization")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene image directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    cfg = RI3DConfig(scene_dir=Path(args.scene),
                     output_dir=Path(args.output) if args.output else None)
    run_stage2(cfg)
