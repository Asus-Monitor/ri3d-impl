"""Step 8: Stage 2 Optimization — inpaint missing regions and finalize.

Per the paper (Sec 4.3, Eq. 4) — refreshed every 200 iters (§8.3):
  1. Render all M novel views
  2. Select K non-overlapping novel views with missing regions (inpaint
     until step < stage2_inpaint_cutoff)
  3. Inpaint missing areas via personalized inpainting model
  4. Project inpainted content into 3D (Poisson-fused mono-depth)
  5. Repair all M views to produce pseudo-GTs
  6. Optimize for stage2_inpaint_interval iters with loss below

  L = sum_i L_rec(rendered_i, gt_i)                        [input views]
    + sum_j λ_j L_rec(rendered_j, repaired_j)              [full image, no M_α]
    + sum_k (1-M_α_k) ⊙ M_b_k ⊙ Lp(rendered_k, inpainted_k) [live M_α]

The live M_α on term 3 is the anti-ratchet: as 3DGS fills in a missing
region, the inpaint supervision naturally deactivates there and term 2
(repair) smoothly takes over for final consistency.

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
from utils import (
    estimate_mono_depth, clear_mono_depth_cache,
    prepare_for_pipeline, load_gt_images, load_mono_depths,
    compute_scene_scale,
)


def load_inpainting_pipeline(cfg: RI3DConfig):
    """Load the trained inpainting model for inference (per-scene fine-tuned UNet)."""
    from diffusers import (
        StableDiffusionInpaintPipeline,
        UNet2DConditionModel,
        DPMSolverMultistepScheduler,
    )

    model_dir = cfg.scene_output_dir() / "inpainting_model"
    dtype = cfg.dtype
    device = cfg.device

    inpaint_model_id = "runwayml/stable-diffusion-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        inpaint_model_id, torch_dtype=dtype
    ).to(device)

    pipe.unet = UNet2DConditionModel.from_pretrained(
        model_dir, torch_dtype=dtype).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True)
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

    img_pil = Image.fromarray(img_np)
    img_resized, pipe_h, pipe_w = prepare_for_pipeline(img_pil)
    mask_pil = Image.fromarray(mask_np, mode="L").resize((pipe_w, pipe_h), Image.NEAREST)

    with torch.no_grad():
        result = pipe(
            prompt="",
            image=img_resized,
            mask_image=mask_pil,
            height=pipe_h,
            width=pipe_w,
            strength=cfg.inpainting_strength,
            num_inference_steps=cfg.inpainting_inference_steps,
            guidance_scale=cfg.inpainting_guidance_scale,
        ).images[0]

    result_np = np.array(result.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
    inpainted = torch.from_numpy(result_np).to(rendered_image.device)

    
    
    missing_3ch = missing.unsqueeze(-1)
    composite = rendered_image * (1 - missing_3ch) + inpainted * missing_3ch

    return composite


def project_inpainted_to_3d(inpainted_image: torch.Tensor, missing_mask: torch.Tensor,
                              rendered_depth: torch.Tensor, c2w: torch.Tensor,
                              K: torch.Tensor, cfg: RI3DConfig,
                              max_new_gaussians: int = 1000) -> dict | None:
    """Project inpainted pixels into 3D as new Gaussians.

    Per paper Sec 4.3 Stage 2: "We address this by combining the monocular
    depth in the inpainted regions with the rendered depth using Eq. 2."
    Uses Poisson fusion (same as step3) with rendered depth as reference
    and visibility mask as the confidence mask.

    Subsamples to max_new_gaussians to prevent unbounded growth. With 14
    inpaint cycles × 5 views, 1000/view caps total inpaint Gaussians at ~70k —
    enough to cover missing regions without superimposing contradictory
    hallucinations from successive cycles.
    """
    from step3_depth_fusion import align_mono_to_dust3r, solve_poisson_fusion_fast

    device = inpainted_image.device
    H, W = inpainted_image.shape[:2]

    
    n_missing = (missing_mask > 0.5).sum().item()
    if n_missing < 100:
        return None

    
    mono_depth = estimate_mono_depth(inpainted_image, cfg)  

    
    visible = (missing_mask < 0.5)
    rd = rendered_depth.squeeze()
    valid = visible & (rd > 0.01)
    if valid.sum() < 100:
        return None

    
    
    rd_np = rd.detach().cpu().numpy().astype(np.float64)
    mono_np = mono_depth.detach().cpu().numpy().astype(np.float64)
    mask_np = valid.detach().cpu().numpy().astype(bool)

    mono_aligned = align_mono_to_dust3r(mono_np, rd_np, mask_np)
    fused_np = solve_poisson_fusion_fast(rd_np, mono_aligned, mask_np, cfg.poisson_lambda)

    
    bad = ~np.isfinite(fused_np)
    if bad.any():
        fused_np[bad] = rd_np[bad]
    fused_np = np.maximum(fused_np, 0.01).astype(np.float32)
    fused = torch.from_numpy(fused_np).to(device)

    
    mask_flat = missing_mask.reshape(-1) > 0.5
    n_total = mask_flat.sum().item()
    if n_total == 0:
        return None

    
    if n_total > max_new_gaussians:
        indices = torch.where(mask_flat)[0]
        perm = torch.randperm(n_total, device=device)[:max_new_gaussians]
        subsample_mask = torch.zeros_like(mask_flat)
        subsample_mask[indices[perm]] = True
        mask_flat = subsample_mask

    
    pts = unproject_depth(fused, K, c2w)
    colors = inpainted_image.reshape(-1, 3)

    pts = pts[mask_flat]
    colors = colors[mask_flat]
    depths_valid = fused.reshape(-1)[mask_flat]

    
    fx = K[0, 0].item()
    pixel_sizes = depths_valid / fx * cfg.gaussian_scale_factor
    scales = pixel_sizes.unsqueeze(-1).expand(-1, 3)

    n = pts.shape[0]
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0
    opacities = torch.full((n,), 0.5, device=device)

    # Convert inpainted RGB to SH DC coefficients (must match existing
    # Gaussians' shape (N, 16, 3) so extend_with_gaussians can concatenate).
    # See step4 for convention: DC = (RGB - 0.5) / C0, higher orders = 0.
    from gsplat.exporter import sh2rgb
    SH_C0 = float(sh2rgb(torch.tensor(1.0)) - sh2rgb(torch.tensor(0.0)))
    K_coefs = 16  # (sh_degree=3 + 1)^2
    sh_coefs = torch.zeros(n, K_coefs, 3, device=device)
    sh_coefs[:, 0, :] = (colors - 0.5) / SH_C0

    return {
        "means": pts,
        "scales": torch.log(scales.clamp(min=1e-8)),
        "quats": quats,
        "opacities": torch.logit(opacities.clamp(1e-4, 1 - 1e-4)),
        "colors": sh_coefs,
    }


def select_inpaint_views(renders_cache: dict, n_views: int, k_inpaint: int,
                          cycle: int) -> list[int]:
    """Select K views for inpainting using alternating even/odd pattern.

    Per paper Sec 8.3: 'sequentially inpaint and project every other view
    (5 views)'.  Alternates between even and odd indices each cycle to ensure
    non-overlapping coverage.  Views with negligible missing regions are skipped.
    """
    if cycle % 2 == 0:
        candidates = list(range(0, n_views, 2))   
    else:
        candidates = list(range(1, n_views, 2))   

    
    selected = []
    for j in candidates:
        if j in renders_cache:
            rc = renders_cache[j]
            missing = ((1 - rc["alpha_mask"]) * rc["bg_mask"]).sum().item()
            if missing > 100:
                selected.append(j)

    return selected[:k_inpaint]


def run_stage2(cfg: RI3DConfig):
    out_dir = cfg.scene_output_dir()
    render_dir = out_dir / "stage2_renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device

    
    image_paths = cfg.load_image_paths()
    poses = torch.load(out_dir / "dust3r_poses.pt", weights_only=True).float().to(device)
    intrinsics = torch.load(out_dir / "dust3r_intrinsics.pt", weights_only=True).float().to(device)
    n_images = len(image_paths)

    
    stage1_ckpt = torch.load(out_dir / "stage1_checkpoint.pt", weights_only=True)
    fused_depth_0 = torch.load(out_dir / "fused_depths" / "fused_depth_000.pt", weights_only=True)
    H, W = fused_depth_0.shape

    
    gt_images = load_gt_images(image_paths, H, W, device)
    mono_depths = load_mono_depths(out_dir, n_images, device)

    
    model = GaussianModel(stage1_ckpt["gaussians"], device)
    optimizers = model.setup_optimizers(cfg)

    scene_scale = compute_scene_scale(poses)
    print(f"  Scene scale: {scene_scale:.4f}")
    strategy, strategy_state = model.setup_strategy(cfg, scene_scale=scene_scale)

    
    ssim_fn = SSIMLoss().to(device)
    lpips_fn = LPIPSLoss(device)

    
    from step4_gaussian_init import compute_scene_center
    scene_center = compute_scene_center(poses, stage1_ckpt["gaussians"]["means"].to(device))
    novel_c2w = generate_elliptical_cameras(poses, cfg.stage2_num_novel_views, scene_center).to(device)
    K_avg = intrinsics.mean(dim=0)

    camera_weights = torch.zeros(cfg.stage2_num_novel_views, device=device)
    for j in range(cfg.stage2_num_novel_views):
        camera_weights[j] = compute_camera_distance_weight(novel_c2w[j], poses)

    
    input_w2c = torch.linalg.inv(poses)       
    novel_w2c = torch.linalg.inv(novel_c2w)   

    
    inpaint_pipe = None
    repair_pipe = None


    pseudo_gt = [None] * cfg.stage2_num_novel_views
    inpainted_images = [None] * cfg.stage2_num_novel_views
    # M_b cached per-refresh (from mono-depth of the current repaired image,
    # same as stage 1). M_α is recomputed live per-step for the inpaint term
    # (paper Eq. 4, term 3 uses current opacity); term 2 has no M_α in stage 2.
    cached_bg_masks = [None] * cfg.stage2_num_novel_views
    inpaint_cycle = 0

    plateau = PlateauDetector(cfg.plateau_window, cfg.plateau_threshold, cfg.plateau_min_iters)
    losses_history = []

    
    
    s2_max = cfg.stage2_repair_strength_max
    s2_min = cfg.stage2_repair_strength_min

    print(f"\n=== Stage 2 Optimization ===")
    print(f"Max iters: {cfg.stage2_max_iters}")
    print(f"Novel views: {cfg.stage2_num_novel_views}, Inpaint views: {cfg.stage2_num_inpaint_views}")
    print(f"Inpaint interval: {cfg.stage2_inpaint_interval}, cutoff: {cfg.stage2_inpaint_cutoff}")
    print(f"Repair strength: {s2_max:.2f} → {s2_min:.2f} (adaptive)")

    for step in tqdm(range(cfg.stage2_max_iters), desc="Stage 2"):

        
        if step % cfg.stage2_inpaint_interval == 0:
            print(f"\n  Refresh cycle at step {step}...")

            with torch.no_grad():
                renders_cache = {}
                for j in range(cfg.stage2_num_novel_views):
                    r = model.render(novel_w2c[j], K_avg, H, W, return_depth=True)
                    alpha_mask = get_opacity_mask(r["alpha"])
                    renders_cache[j] = {
                        "image": r["image"].detach(),
                        "depth": r["depth"].detach(),
                        "alpha": r["alpha"].detach(),
                        "alpha_mask": alpha_mask,
                    }

                
                
                
                
                for j in range(cfg.stage2_num_novel_views):
                    inpainted_images[j] = None

                
                
                render_bg_masks = {}
                for j in range(cfg.stage2_num_novel_views):
                    rc = renders_cache[j]
                    render_bg_masks[j] = compute_background_mask(
                        rc["depth"].squeeze(-1).detach(), cfg.bg_mask_n_clusters
                    ).to(device)

                do_inpaint = step < cfg.stage2_inpaint_cutoff
                if do_inpaint:
                    
                    
                    for j in render_bg_masks:
                        renders_cache[j]["bg_mask"] = render_bg_masks[j]
                    inpaint_views = select_inpaint_views(
                        renders_cache, cfg.stage2_num_novel_views,
                        cfg.stage2_num_inpaint_views, inpaint_cycle,
                    )
                    inpaint_cycle += 1

                    if inpaint_views:
                        
                        if inpaint_pipe is None:
                            print("  Loading inpainting pipeline...")
                            inpaint_pipe = load_inpainting_pipeline(cfg)

                        for j in inpaint_views:
                            rc = renders_cache[j]
                            inpainted = inpaint_missing_regions(
                                inpaint_pipe, rc["image"], rc["alpha_mask"],
                                render_bg_masks[j], cfg,
                            )
                            inpainted_images[j] = inpainted.clamp(0, 1).detach()

                            
                            missing_mask = ((1 - rc["alpha_mask"]) * render_bg_masks[j]).detach()
                            new_gs = project_inpainted_to_3d(
                                inpainted, missing_mask,
                                rc["depth"], novel_c2w[j], K_avg, cfg,
                            )
                            if new_gs is not None:
                                n_before = model.n_gaussians
                                
                                model.extend_with_gaussians(new_gs, cfg)
                                optimizers = model._optimizers
                                strategy_state = model._strategy_state
                                print(f"    View {j}: +{model.n_gaussians - n_before} "
                                      f"Gaussians (total: {model.n_gaussians})")
                else:
                    
                    if inpaint_pipe is not None:
                        del inpaint_pipe
                        inpaint_pipe = None
                        clear_mono_depth_cache()
                        torch.cuda.empty_cache()
                        print("  Freed inpainting + depth models (past cutoff)")

                
                
                
                if repair_pipe is None:
                    print("  Loading repair pipeline...")
                    repair_pipe = load_repair_pipeline(cfg)

                
                progress = step / max(cfg.stage2_max_iters - 1, 1)
                cur_strength = s2_max + (s2_min - s2_max) * progress

                for j in range(cfg.stage2_num_novel_views):
                    rc = renders_cache[j]
                    if inpainted_images[j] is not None:
                        repaired = repair_image(repair_pipe, inpainted_images[j], cfg,
                                                view_index=j, strength_override=cur_strength)
                    else:
                        repaired = repair_image(repair_pipe, rc["image"], cfg,
                                                view_index=j, strength_override=cur_strength)
                    pseudo_gt[j] = repaired.clamp(0, 1).detach()

                    
                    
                    
                    mono_d = estimate_mono_depth(repaired, cfg)
                    cached_bg_masks[j] = compute_background_mask(
                        mono_d, cfg.bg_mask_n_clusters
                    ).to(device)

                torch.cuda.empty_cache()

            
            if step % (cfg.stage2_inpaint_interval * 4) == 0:
                for j in range(min(3, cfg.stage2_num_novel_views)):
                    rc = renders_cache[j]
                    n_cols = 4 if inpainted_images[j] is not None else 3
                    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
                    axes[0].imshow(rc["image"].clamp(0, 1).cpu().numpy())
                    axes[0].set_title(f"Rendered (step {step})")
                    axes[1].imshow(pseudo_gt[j].cpu().numpy())
                    axes[1].set_title("Pseudo GT")
                    axes[2].imshow(rc["alpha"].squeeze(-1).cpu().numpy(), cmap="gray")
                    axes[2].set_title("Alpha")
                    if inpainted_images[j] is not None:
                        axes[3].imshow(inpainted_images[j].cpu().numpy())
                        axes[3].set_title("Inpainted")
                    for ax in axes: ax.axis("off")
                    fig.savefig(render_dir / f"step{step:05d}_novel{j}.png",
                                dpi=120, bbox_inches="tight")
                    plt.close(fig)

            del renders_cache

        
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

        
        
        
        # Novel-view terms (Eq. 4): all M views with per-paper weights λ_j.
        # Term 2 in stage 2 has NO M_α mask (paper: "we enforce the loss in
        # the second term across the entire image by removing the visibility
        # mask"). Term 3 uses LIVE M_α from current render.
        if pseudo_gt[0] is not None:
            n_nov = cfg.stage2_num_novel_views
            for nov_idx in range(n_nov):
                w2c_nov = novel_w2c[nov_idx]

                result_nov = model.render_for_loss(w2c_nov, K_avg, H, W, render_mode="RGB")
                lambda_j = camera_weights[nov_idx]

                view_loss = lambda_j * reconstruction_loss(
                    result_nov["image"], pseudo_gt[nov_idx],
                    ssim_fn, lpips_fn_or_none, cfg,
                )

                if inpainted_images[nov_idx] is not None and cached_bg_masks[nov_idx] is not None:
                    # Live M_α — as 3DGS fills in, the inpaint loss naturally
                    # deactivates and term 2 (repair) takes over. This is the
                    # paper's anti-ratchet in stage 2: supervision transitions
                    # from "pull content into this hole" to "keep rendered
                    # content consistent with the repaired image".
                    alpha_mask_live = get_opacity_mask(result_nov["alpha"]).detach()
                    mask_ip = (1 - alpha_mask_live) * cached_bg_masks[nov_idx]
                    if mask_ip.sum() > 100:
                        mask_ip_3ch = mask_ip.unsqueeze(-1)
                        n_ip_pixels = mask_ip.sum().clamp(min=1)

                        ip_l1 = ((result_nov["image"] - inpainted_images[nov_idx]).abs() * mask_ip_3ch).sum() / (n_ip_pixels * 3)
                        ip_loss = cfg.loss_l1_weight * ip_l1

                        if lpips_fn_or_none is not None:
                            lpips_map = lpips_fn(result_nov["image"], inpainted_images[nov_idx],
                                                 return_map=True)
                            if lpips_map.shape != mask_ip.shape:
                                lpips_map = F.interpolate(
                                    lpips_map.unsqueeze(0).unsqueeze(0),
                                    size=mask_ip.shape, mode="bilinear", align_corners=False,
                                ).squeeze()
                            ip_lpips = (lpips_map * mask_ip).sum() / n_ip_pixels
                            ip_loss = ip_loss + cfg.loss_lpips_weight * ip_lpips
                        view_loss = view_loss + ip_loss

                view_loss.backward()
                loss_val += view_loss.item()

        
        model.optimizer_step()
        losses_history.append(loss_val)

        if (step + 1) % 200 == 0:
            print(f"  Step {step+1}: loss = {loss_val:.6f}, "
                  f"n_gaussians = {model.n_gaussians}")

        if plateau.update(loss_val):
            print(f"\n  Plateau detected at step {step+1}, stopping Stage 2.")
            break

    
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

    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses_history)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Stage 2 Loss")
    fig.savefig(out_dir / "stage2_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    
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
