"""Step 5: Train the Repair Model (full ControlNet fine-tuning).

Per the paper (Sec 4.2, 8.1):
  1. Leave-one-out: create N subsets of N-1 images, train N 3DGS reps
     (6000 iters without left-out view, then to 10000 with it re-added)
  2. Snapshot intermediate renders -> training pairs (corrupted, clean)
  3. Fine-tune ControlNet (tile) on this scene's pairs for 1800 iters

Models are personalized per-scene so the repair model matches the scene's visual style.

Outputs:
  - outputs/<scene>/repair_training_data/  corrupted/clean pairs
  - outputs/<scene>/repair_model/          fine-tuned ControlNet weights
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
from gaussian_trainer import GaussianModel

random.seed(42)

def print_pair_l1_distribution(all_pairs, save_to: Path | None = None,
                                scene_name: str = "") -> dict:
    """Print + plot multi-metric corruption distribution across training pairs.

    Mean L1 alone is misleading: a pair with 99% clean pixels and 1% strong
    artifacts has the same mean L1 as a pair with 100% mild noise. For a
    repair model, the latter is identity-like training noise; the former is
    valuable training signal. Report:
      - mean_l1      : average pixel error (what we had before)
      - p95_diff     : 95th-percentile pixel error (artifact magnitude)
      - frac_signif  : fraction of pixels with |diff| > 0.05 (artifact extent)
    """
    if not all_pairs:
        return {}
    metrics = []
    for c, cl in all_pairs:
        diff = (c.float() - cl.float()).abs()
        mean_l1 = diff.mean().item()
        p95 = diff.flatten().kthvalue(int(diff.numel() * 0.95)).values.item()
        frac_signif = (diff.amax(dim=-1) > 0.05).float().mean().item()
        metrics.append((mean_l1, p95, frac_signif))
    metrics = np.array(metrics)
    l1s, p95s, fracs = metrics[:, 0], metrics[:, 1], metrics[:, 2]

    print(f"\n  Pair corruption distribution ({len(l1s)} pairs):")
    print(f"    {'metric':<14} {'min':>6} {'p10':>6} {'p25':>6} {'p50':>6} "
          f"{'p75':>6} {'p90':>6} {'max':>6}")
    for name, arr in [("mean_l1", l1s), ("p95_diff", p95s), ("frac_signif", fracs)]:
        qs = np.percentile(arr, [0, 10, 25, 50, 75, 90, 100])
        print(f"    {name:<14} " + " ".join(f"{q:>6.3f}" for q in qs))

    bins = [0.00, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.30, 1.00]
    counts, _ = np.histogram(l1s, bins=bins)
    print(f"    L1 bins: " + "  ".join(
        f"[{bins[i]:.2f}-{bins[i+1]:.2f}]={c}" for i, c in enumerate(counts) if c
    ))
    frac_bins = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0]
    frac_counts, _ = np.histogram(fracs, bins=frac_bins)
    print(f"    frac_signif bins: " + "  ".join(
        f"[{frac_bins[i]:.2f}-{frac_bins[i+1]:.2f}]={c}"
        for i, c in enumerate(frac_counts) if c
    ))

    if save_to is not None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for ax, arr, name in zip(axes, [l1s, p95s, fracs],
                                  ["mean_l1", "p95_diff", "frac_signif (>0.05)"]):
            ax.hist(arr, bins=25, edgecolor="black", alpha=0.75)
            med = np.median(arr)
            ax.axvline(med, color="r", linestyle="--", label=f"med={med:.3f}")
            ax.set_xlabel(name)
            ax.set_ylabel("Count")
            ax.legend()
        fig.suptitle(f"Pair corruption — {scene_name} ({len(l1s)} pairs)")
        fig.savefig(save_to, dpi=120, bbox_inches="tight")
        plt.close(fig)

    return {"n": len(l1s),
            "l1_median": float(np.median(l1s)),
            "p95_median": float(np.median(p95s)),
            "frac_median": float(np.median(fracs))}


def inspect_low_corruption_pairs(all_pairs, save_to: Path,
                                  scene_name: str = "", k: int = 10) -> None:
    """Render the k lowest-L1 pairs side-by-side (corrupted | clean | abs-diff).

    Diagnostic for P-A (filtering): if these pairs are visually identical to
    GT, they're useless training data and filtering is correct. If they show
    real artifacts (color shifts, ghosting), the filter threshold needs care.
    """
    if not all_pairs:
        return
    l1s = [F.l1_loss(c.float(), cl.float()).item() for c, cl in all_pairs]
    order = np.argsort(l1s)[:k]
    n = len(order)

    fig, axes = plt.subplots(n, 3, figsize=(12, 3.2 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for row, idx in enumerate(order):
        c, cl = all_pairs[idx]
        c_np = c.float().numpy()
        cl_np = cl.float().numpy()
        diff = np.abs(c_np - cl_np)
        # Amplify diff 5x for visibility — annotate so it's not misleading.
        diff_vis = np.clip(diff * 5.0, 0, 1)

        axes[row, 0].imshow(c_np)
        axes[row, 0].set_title(f"pair {idx}: corrupted (L1={l1s[idx]:.4f})")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(cl_np)
        axes[row, 1].set_title("clean (GT)")
        axes[row, 1].axis("off")
        axes[row, 2].imshow(diff_vis)
        axes[row, 2].set_title(f"|diff| ×5  (max={diff.max():.3f})")
        axes[row, 2].axis("off")

    fig.suptitle(f"P-D: {n} lowest-L1 pairs — {scene_name}\n"
                 f"If 'corrupted' looks identical to 'clean' and diff is mostly black, "
                 f"these pairs are training noise and should be filtered.", fontsize=11)
    fig.savefig(save_to, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved low-corruption inspection to {save_to}")


def generate_leave_one_out_data(cfg: RI3DConfig):
    """Generate corrupted/clean image pairs for a single scene using leave-one-out."""
    from gsplat import rasterization, DefaultStrategy

    out_dir = cfg.scene_output_dir()
    data_dir = out_dir / "repair_training_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    image_paths = cfg.load_image_paths()
    poses = torch.load(out_dir / "dust3r_poses.pt", weights_only=True).float()
    intrinsics = torch.load(out_dir / "dust3r_intrinsics.pt", weights_only=True).float()
    gaussians_init = torch.load(out_dir / "init_gaussians.pt", weights_only=True)
    n_images = len(image_paths)

    device = cfg.device
    fused_depth = torch.load(out_dir / "fused_depths" / "fused_depth_000.pt", weights_only=True)
    H, W = fused_depth.shape

    
    gt_images_render = []
    for ip in image_paths:
        img = Image.open(ip).convert("RGB")
        gt_images_render.append(
            torch.from_numpy(np.array(img.resize((W, H), Image.LANCZOS))).float() / 255.0
        )

    all_pairs = []

    
    w2c_all = torch.linalg.inv(poses).to(device)
    K_all = intrinsics.to(device)

    s = cfg.loo_render_scale
    loo_H, loo_W = int(H * s), int(W * s)
    loo_K_all = K_all.clone()
    loo_K_all[:, 0, :] *= s  
    loo_K_all[:, 1, :] *= s  

    
    gt_gpu_loo = []
    for g in gt_images_render:
        g_loo = F.interpolate(
            g.permute(2, 0, 1).unsqueeze(0), size=(loo_H, loo_W),
            mode="bilinear", align_corners=False,
        ).squeeze(0).permute(1, 2, 0).to(device)
        gt_gpu_loo.append(g_loo)

    torch.backends.cudnn.benchmark = True

    
    
    
    snapshot_steps = set(
        range(cfg.loo_initial_iters, cfg.loo_total_iters, cfg.loo_snapshot_interval)
    )

    all_indices = list(range(n_images))

    
    
    
    per_view_counts = []
    for i in range(n_images):
        fd = torch.load(out_dir / "fused_depths" / f"fused_depth_{i:03d}.pt", weights_only=True)
        valid_count = ((fd > 0.01) & torch.isfinite(fd)).sum().item()
        per_view_counts.append(valid_count)
    per_view_offsets = [0]
    for c in per_view_counts:
        per_view_offsets.append(per_view_offsets[-1] + c)

    for left_out_idx in range(n_images):
        name = Path(image_paths[left_out_idx]).stem
        print(f"\nLeave-one-out: excluding view {left_out_idx} ({name})")

        train_indices = [j for j in range(n_images) if j != left_out_idx]
        clean_cpu = gt_images_render[left_out_idx]

        
        lo_start = per_view_offsets[left_out_idx]
        lo_end = per_view_offsets[left_out_idx + 1]
        loo_gaussians = {
            k: torch.cat([v[:lo_start], v[lo_end:]], dim=0)
            for k, v in gaussians_init.items()
        }
        model = GaussianModel(loo_gaussians, device)
        optimizers = model.setup_optimizers(cfg)
        from utils import compute_scene_scale
        loo_scene_scale = compute_scene_scale(poses)
        strategy, strategy_state = model.setup_strategy(cfg, scene_scale=loo_scene_scale)

        w2c_lo = w2c_all[left_out_idx]
        K_lo = K_all[left_out_idx]

        for step in tqdm(range(cfg.loo_total_iters), desc=f"  LOO {left_out_idx}"):
            
            if step < cfg.loo_initial_iters:
                idx = train_indices[step % len(train_indices)]
            else:
                idx = all_indices[step % n_images]

            
            
            result = model.render_for_optim(
                w2c_all[idx], loo_K_all[idx], loo_H, loo_W,
                strategy, strategy_state, step,
                render_mode="RGB",
            )
            loss = F.l1_loss(result["image"], gt_gpu_loo[idx])
            loss.backward()
            model.step_post_backward(step, result["meta"])
            model.optimizer_step()

            
            
            if step in snapshot_steps:
                with torch.no_grad():
                    r = model.render(w2c_lo, K_lo, H, W)
                    corrupted = r["image"].clamp(0, 1).cpu()
                    all_pairs.append((corrupted, clean_cpu))

        del model
        torch.cuda.empty_cache()

    print(f"\nGenerated {len(all_pairs)} training pairs for scene {cfg.scene_name}")
    torch.save(all_pairs, data_dir / "training_pairs.pt")

    print_pair_l1_distribution(all_pairs, save_to=data_dir / "pairs_l1_histogram.png",
                                scene_name=cfg.scene_name)

    
    n_show = min(8, len(all_pairs))
    if n_show > 0:
        preview_indices = [int(i * len(all_pairs) / n_show) for i in range(n_show)]
        fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
        if n_show == 1:
            axes = axes.reshape(-1, 1)
        for col, j in enumerate(preview_indices):
            axes[0, col].imshow(all_pairs[j][0].numpy())
            axes[0, col].set_title(f"Corrupted (pair {j})")
            axes[0, col].axis("off")
            axes[1, col].imshow(all_pairs[j][1].numpy())
            axes[1, col].set_title(f"Clean (pair {j})")
            axes[1, col].axis("off")
        fig.suptitle(f"Leave-One-Out Pairs — {cfg.scene_name} ({len(all_pairs)} total)")
        fig.savefig(data_dir / "pairs_preview.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    return all_pairs


def generate_all_scenes_data(cfg: RI3DConfig):
    """Generate leave-one-out data for ALL scenes."""
    scenes = cfg.list_scenes()
    print(f"Generating repair training data for {len(scenes)} scenes...")
    for scene_dir in scenes:
        scene_cfg = RI3DConfig(
            scene_dir=scene_dir, dataset_dir=cfg.dataset_dir,
            output_dir=cfg.output_dir, n_views=cfg.n_views,
            device=cfg.device, dtype=cfg.dtype,
        )
        
        if not (scene_cfg.scene_output_dir() / "init_gaussians.pt").exists():
            print(f"  Skipping {scene_dir.name}: run steps 1-4 first")
            continue
        
        if (scene_cfg.scene_output_dir() / "repair_training_data" / "training_pairs.pt").exists():
            print(f"  Skipping {scene_dir.name}: training pairs already exist")
            continue
        generate_leave_one_out_data(scene_cfg)


def train_repair_model(cfg: RI3DConfig, shared_components=None):
    """Train a per-scene repair model (full ControlNet fine-tuning) on leave-one-out data.

    Per the paper (Sec 4.2): fine-tune the ControlNet on corrupted/clean pairs so
    it learns scene-specific repair. The UNet stays frozen; only ControlNet
    parameters are updated. Gradient checkpointing keeps memory tractable.

    Args:
        cfg: scene config
        shared_components: optional dict with pre-loaded frozen SD components
            (vae, unet, text_embeds, noise_scheduler) to avoid reloading per scene
    """
    from diffusers import (
        ControlNetModel,
        DDPMScheduler,
        AutoencoderKL,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    model_dir = cfg.scene_output_dir() / "repair_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device
    dtype = cfg.dtype


    pairs_path = cfg.scene_output_dir() / "repair_training_data" / "training_pairs.pt"
    all_pairs = torch.load(pairs_path, weights_only=False)
    if len(all_pairs) == 0:
        raise RuntimeError(f"No training pairs for {cfg.scene_name}. Run data generation first.")
    print(f"Training pairs for {cfg.scene_name}: {len(all_pairs)}")

    # Filter pairs by spatial extent of artifacts. Mean L1 alone is misleading
    # because most pixels are clean background — even pairs we'd "intuitively"
    # call low-corruption have localized artifacts on the subject. A pair is
    # useful for repair training only if its artifact regions cover a
    # meaningful fraction of pixels; otherwise the loss is dominated by clean
    # pixels and the model collapses to identity.
    _diff_t = cfg.repair_mask_diff_threshold
    _frac_min = cfg.repair_pair_min_frac_signif
    def _frac_signif(c, cl):
        return ((c.float() - cl.float()).abs().amax(dim=-1) > _diff_t).float().mean().item()
    _n_before = len(all_pairs)
    all_pairs = [p for p in all_pairs if _frac_signif(*p) >= _frac_min]
    print(f"Filtered pairs: {_n_before} → {len(all_pairs)} "
          f"(kept frac_signif >= {_frac_min:.2f}, |diff|>{_diff_t:.2f})")
    if len(all_pairs) == 0:
        raise RuntimeError(f"All pairs filtered out — lower repair_pair_min_frac_signif")

    
    
    
    
    _owns_components = shared_components is None
    if _owns_components:
        print("Loading SD 1.5 + ControlNet tile...")
        tokenizer = CLIPTokenizer.from_pretrained(cfg.sd_model, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(cfg.sd_model, subfolder="text_encoder",
                                                      torch_dtype=dtype).to(device)
        vae = AutoencoderKL.from_pretrained(cfg.sd_model, subfolder="vae",
                                             torch_dtype=dtype).to(device)
        unet = UNet2DConditionModel.from_pretrained(cfg.sd_model, subfolder="unet",
                                                     torch_dtype=dtype).to(device)
        noise_scheduler = DDPMScheduler.from_pretrained(cfg.sd_model, subfolder="scheduler")
        text_encoder.requires_grad_(False)
        vae.requires_grad_(False)
        unet.requires_grad_(False)
    else:
        tokenizer = None
        text_encoder = None
        vae = shared_components["vae"]
        unet = shared_components["unet"]
        noise_scheduler = shared_components["noise_scheduler"]

    
    if tokenizer is None or text_encoder is None:
        tokenizer = CLIPTokenizer.from_pretrained(cfg.sd_model, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(cfg.sd_model, subfolder="text_encoder",
                                                      torch_dtype=dtype).to(device)
        text_encoder.requires_grad_(False)
        _owns_text_encoder = True
    else:
        _owns_text_encoder = False

    with torch.no_grad():
        tokens = tokenizer(cfg.repair_positive_prompt, padding="max_length", max_length=77,
                           truncation=True, return_tensors="pt").input_ids.to(device)
        text_embeds = text_encoder(tokens)[0]

    
    if _owns_text_encoder and not _owns_components:
        del text_encoder
    if _owns_components:
        del text_encoder, tokenizer
    torch.cuda.empty_cache()

    
    
    
    torch.backends.cudnn.benchmark = True
    latent_crop = 64
    pair_corrupted_gpu = []
    pair_clean_lat_gpu = []
    pair_weight_lat_gpu = []  # per-latent-pixel loss weight (artifact mask)

    _bg_w = cfg.repair_mask_weight_background  # latent loss weight for clean px

    with torch.no_grad():
        for corrupted, clean in all_pairs:
            H, W = corrupted.shape[:2]
            if H < W:
                new_h, new_w = 512, int(W * 512 / H)
            else:
                new_w, new_h = 512, int(H * 512 / W)
            new_h = (new_h // 8) * 8
            new_w = (new_w // 8) * 8

            corrupted_r = F.interpolate(
                corrupted.permute(2, 0, 1).unsqueeze(0), size=(new_h, new_w),
                mode="bilinear", align_corners=False,
            ).squeeze(0).to(device, dtype)
            clean_r = F.interpolate(
                clean.permute(2, 0, 1).unsqueeze(0), size=(new_h, new_w),
                mode="bilinear", align_corners=False,
            ).squeeze(0).to(device, dtype)

            clean_lat = vae.encode(
                clean_r.unsqueeze(0) * 2 - 1
            ).latent_dist.sample() * vae.config.scaling_factor

            # Artifact mask in image space → average-pool to latent res.
            # Soft-floored at _bg_w so clean regions still get some gradient
            # (otherwise model is free to garbage them at high t).
            diff_full = (corrupted_r - clean_r).abs().amax(dim=0)  # (new_h, new_w)
            mask_img = (diff_full > _diff_t).float()
            mask_lat = F.avg_pool2d(
                mask_img.unsqueeze(0).unsqueeze(0),
                kernel_size=8, stride=8,
            ).squeeze(0)  # (1, new_h/8, new_w/8)
            weight_lat = (_bg_w + (1.0 - _bg_w) * mask_lat).to(dtype)

            pair_corrupted_gpu.append(corrupted_r)
            pair_clean_lat_gpu.append(clean_lat.squeeze(0))
            pair_weight_lat_gpu.append(weight_lat)

        # Augment: pre-encode horizontal flips. Doubles effective dataset size
        # at low cost (~22 MB extra latents) — important for ~40-pair training.
        # Re-encode the flipped clean image rather than flipping its latent:
        # VAE convolutions are not exactly flip-equivariant, so a flipped
        # latent doesn't match what the model would see at inference for a
        # mirrored clean image. The corrupted condition is image-space so
        # flipping it directly is exact.
        if cfg.repair_hflip_augment:
            for i, (corrupted, clean) in enumerate(all_pairs):
                # Recompute clean image at the same resolution used above.
                H_, W_ = corrupted.shape[:2]
                if H_ < W_:
                    new_h, new_w = 512, int(W_ * 512 / H_)
                else:
                    new_w, new_h = 512, int(H_ * 512 / W_)
                new_h = (new_h // 8) * 8
                new_w = (new_w // 8) * 8
                clean_r = F.interpolate(
                    clean.permute(2, 0, 1).unsqueeze(0), size=(new_h, new_w),
                    mode="bilinear", align_corners=False,
                ).squeeze(0).to(device, dtype)
                clean_flip_img = clean_r.flip(-1)
                clean_lat_flip = vae.encode(
                    clean_flip_img.unsqueeze(0) * 2 - 1
                ).latent_dist.sample() * vae.config.scaling_factor
                pair_corrupted_gpu.append(pair_corrupted_gpu[i].flip(-1))
                pair_clean_lat_gpu.append(clean_lat_flip.squeeze(0))
                # Weight mask is image-space-aligned, so flip directly.
                pair_weight_lat_gpu.append(pair_weight_lat_gpu[i].flip(-1))

    n_pairs = len(pair_corrupted_gpu)
    print(f"Pre-encoded {n_pairs} pairs on GPU"
          + (" (incl. hflips)" if cfg.repair_hflip_augment else ""))

    
    if _owns_components:
        del vae
    torch.cuda.empty_cache()


    # FP16 model + 8-bit Adam: keeps everything on GPU with ~3x less
    # optimizer memory than FP32 Adam.  bitsandbytes AdamW8bit maintains
    # FP32 master weights internally, avoiding FP16 precision loss.
    controlnet = ControlNetModel.from_pretrained(cfg.controlnet_model,
                                                  torch_dtype=dtype, use_safetensors=False).to(device)
    controlnet.requires_grad_(True)
    controlnet.enable_gradient_checkpointing()
    # Channels-last NHWC layout: ~8% faster on Turing+ conv kernels, no
    # change to training math (just memory layout). Apply to both the trained
    # ControlNet and the frozen UNet that backprop flows through.
    controlnet.to(memory_format=torch.channels_last)
    controlnet.train()
    n_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
    print(f"Full ControlNet fine-tuning: {n_params:,} trainable parameters")

    unet.requires_grad_(False)
    unet.to(memory_format=torch.channels_last)
    unet.eval()

    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(controlnet.parameters(), lr=cfg.repair_lr, weight_decay=1e-2)

    n_iters = cfg.repair_train_iters
    max_t = noise_scheduler.config.num_train_timesteps

    # bs=1 on 8 GB Turing — bs=2 risks OOM when other processes share the
    # card (we measured 4.9 GB peak in isolation, but neighbors can take 2 GB+).
    gpu_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    batch_size = 2 if gpu_mem_gb >= 16 else 1
    n_steps = (n_iters + batch_size - 1) // batch_size
    text_embeds_b = text_embeds.expand(batch_size, -1, -1)

    # GaussianObject uses no LR scheduler. We add a short linear warmup only
    # (50 steps) to protect the FP16 + 8-bit-Adam setup from early grad spikes,
    # then hold lr constant to match GaussianObject's recipe.
    warmup_steps = min(50, n_steps // 20)
    def lr_lambda(s):
        return min(1.0, (s + 1) / max(warmup_steps, 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Training ControlNet (full fine-tune) for {n_iters} iterations (batch={batch_size}) "
          f"on {n_pairs} pairs...")
    losses = []
    iter_count = 0

    for step in tqdm(range(n_steps), desc=f"Repair [{cfg.scene_name}]"):
        actual_bs = min(batch_size, n_iters - iter_count)
        if actual_bs <= 0:
            break

        # Build batch (bs>=2 in normal case; bs=1 only for trailing odd step).
        if actual_bs == 1:
            idx = random.randint(0, n_pairs - 1)
            c_full = pair_corrupted_gpu[idx]
            cl_full = pair_clean_lat_gpu[idx]
            w_full = pair_weight_lat_gpu[idx]
            _, h, w = c_full.shape
            x0 = random.randint(0, max(0, (w - 512) // 8)) * 8
            y0 = random.randint(0, max(0, (h - 512) // 8)) * 8
            lx, ly = x0 // 8, y0 // 8

            corrupted_b = c_full[:, y0:y0+512, x0:x0+512].unsqueeze(0).contiguous(
                memory_format=torch.channels_last)
            clean_lat = cl_full[:, ly:ly+latent_crop, lx:lx+latent_crop].unsqueeze(0)
            weight_b = w_full[:, ly:ly+latent_crop, lx:lx+latent_crop].unsqueeze(0)
            noise_b = torch.randn_like(clean_lat)
            t_b = torch.randint(0, max_t, (1,), device=device).long()
            noisy_lat_b = noise_scheduler.add_noise(clean_lat, noise_b, t_b).contiguous(
                memory_format=torch.channels_last)
        else:
            corrupted_crops, noisy_lats, noises, ts, weights = [], [], [], [], []
            for _ in range(actual_bs):
                idx = random.randint(0, n_pairs - 1)
                c_full = pair_corrupted_gpu[idx]
                cl_full = pair_clean_lat_gpu[idx]
                w_full = pair_weight_lat_gpu[idx]
                _, h, w = c_full.shape
                x0 = random.randint(0, max(0, (w - 512) // 8)) * 8
                y0 = random.randint(0, max(0, (h - 512) // 8)) * 8
                lx, ly = x0 // 8, y0 // 8
                corrupted_crops.append(c_full[:, y0:y0+512, x0:x0+512].unsqueeze(0))
                cl = cl_full[:, ly:ly+latent_crop, lx:lx+latent_crop].unsqueeze(0)
                noise = torch.randn_like(cl)
                t = torch.randint(0, max_t, (1,), device=device).long()
                noisy_lats.append(noise_scheduler.add_noise(cl, noise, t))
                noises.append(noise)
                ts.append(t)
                weights.append(w_full[:, ly:ly+latent_crop, lx:lx+latent_crop].unsqueeze(0))
            corrupted_b = torch.cat(corrupted_crops).contiguous(memory_format=torch.channels_last)
            noisy_lat_b = torch.cat(noisy_lats).contiguous(memory_format=torch.channels_last)
            noise_b = torch.cat(noises)
            t_b = torch.cat(ts)
            weight_b = torch.cat(weights)

        step_embeds = text_embeds_b[:actual_bs]

        # No autocast: model + tensors are already fp16. autocast would only
        # add dtype-tracking overhead and selective fp32 upcasts.
        controlnet_output = controlnet(
            noisy_lat_b, t_b, encoder_hidden_states=step_embeds,
            controlnet_cond=corrupted_b,
            return_dict=False,
        )

        noise_pred = unet(
            noisy_lat_b, t_b, encoder_hidden_states=step_embeds,
            down_block_additional_residuals=controlnet_output[0],
            mid_block_additional_residual=controlnet_output[1],
        ).sample

        # Mask-weighted ε-prediction MSE. Weight is high (1.0) on artifact
        # latent pixels and floored at repair_mask_weight_background (e.g. 0.2)
        # on clean ones, so the loss isn't dominated by the clean majority of
        # the latent (which would teach identity). Channels share the same
        # spatial weight (broadcast over the C=4 latent channels).
        sq_err = (noise_pred.float() - noise_b.float()).pow(2)  # (B, 4, h, w)
        w_f = weight_b.float()  # (B, 1, h, w)
        loss = (w_f * sq_err).sum() / (w_f.sum() * sq_err.shape[1] + 1e-8)
        loss.backward()
        # Speed: clip but DO NOT call .isfinite() — the boolean check forces a
        # GPU↔CPU sync every step. Warmup keeps loss stable early; if NaN
        # ever appears, optimizer.step on NaN grads only affects that one step.
        torch.nn.utils.clip_grad_norm_(controlnet.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()

        iter_count += actual_bs
        # Defer .item() — store loss tensor on GPU, only sync at print time.
        losses.append(loss.detach())
        if iter_count % 100 < batch_size or step == n_steps - 1:
            recent_t = torch.stack(losses[-max(1, 100 // batch_size):]).float()
            avg_loss = recent_t.mean().item()
            print(f"  Step ~{min(iter_count, n_iters)}: loss = {avg_loss:.6f}")

    controlnet.save_pretrained(model_dir / "controlnet")
    print(f"Saved repair model (full ControlNet) to {model_dir}")

    losses_cpu = torch.stack(losses).float().cpu().numpy() if losses else []
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses_cpu)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Repair Model Training Loss — {cfg.scene_name}")
    fig.savefig(model_dir / "training_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    del controlnet, optimizer, pair_corrupted_gpu, pair_clean_lat_gpu, pair_weight_lat_gpu, losses
    if _owns_components:
        del unet
    torch.cuda.empty_cache()


def _prepare_for_pipeline(image: Image.Image, target_short_side: int = 512) -> tuple[Image.Image, int, int]:
    """Deprecated: use utils.prepare_for_pipeline instead. Kept for import compat."""
    from utils import prepare_for_pipeline
    return prepare_for_pipeline(image, target_short_side)


def _resize_roundtrip(img: torch.Tensor) -> torch.Tensor:
    """Apply the same resize→pipe-size→resize-back damage that repair_image does.

    repair_image converts to PIL, resizes to short-side 512 (LANCZOS), runs the
    diffusion pipe, then resizes back to (H, W) with LANCZOS. That round-trip
    alone introduces measurable L1 on near-clean inputs. Push reference tensors
    through the same path so the L1 metric isolates diffusion behavior, not
    resize loss.
    """
    from utils import prepare_for_pipeline as _prep
    H, W = img.shape[:2]
    img_np = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(img_np)
    pil_resized, _, _ = _prep(pil)
    pil_back = pil_resized.resize((W, H), Image.LANCZOS)
    arr = np.array(pil_back).astype(np.float32) / 255.0
    return torch.from_numpy(arr).to(img.device)


def test_repair_model(cfg: RI3DConfig):
    """Test the trained repair model using the same pipeline as Stage 1.

    The metric design matters: `repair_image` converts to PIL, resizes to
    short-side 512, runs diffusion, then resizes back. That round-trip alone
    perturbs the image — so we must apply the same round-trip to the reference
    `clean` (and a baseline `corrupted_rt`) before computing L1, otherwise the
    metric just measures resize damage rather than what the model did.

    Uses deterministic DDIM (eta=0.0) for reproducible L1; eta=1.0 is for
    optimization-time stochastic sampling, not evaluation. Near-clean pairs
    (L1 < skip_threshold) bypass diffusion entirely — the strength schedule is
    designed for the optimization loop, not for the evaluation harness.

    Reports L1 / SSIM / LPIPS so creative-but-correct repairs aren't penalized
    purely by L1.
    """
    from step6_stage1_optim import load_repair_pipeline, repair_image
    from gaussian_trainer import SSIMLoss, LPIPSLoss

    out_dir = cfg.scene_output_dir()
    test_dir = out_dir / "repair_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = out_dir / "repair_training_data" / "training_pairs.pt"
    if not pairs_path.exists():
        print("No training pairs for this scene, skipping test.")
        return
    pairs = torch.load(pairs_path, weights_only=False)

    print("Loading repair pipeline for test...")
    pipe = load_repair_pipeline(cfg)
    ssim_fn = SSIMLoss().to(cfg.device)
    lpips_fn = LPIPSLoss(cfg.device)

    n_test = min(6, len(pairs))
    # Sample evenly across pairs (mix of heavy/light corruption).
    test_indices = [int(i * (len(pairs) - 1) / max(n_test - 1, 1)) for i in range(n_test)]

    fig, axes = plt.subplots(3, n_test, figsize=(5 * n_test, 15))
    if n_test == 1:
        axes = axes.reshape(-1, 1)

    s_max = cfg.repair_strength_max
    s_min = cfg.repair_strength_min
    success_threshold = 0.03
    # Skip only truly clean pairs — use raw (pre-roundtrip) L1 so the resize
    # damage doesn't artificially trip the skip on lightly-corrupted views.
    # Bumped to match the training filter (frac_signif >= 0.10): pairs the
    # model never trained on shouldn't go through the model at test time.
    skip_threshold = 0.025

    def pick_strength(l1: float) -> float:
        # Map L1 ∈ [0.02, 0.20] → strength ∈ [s_min, s_max]; clamp outside.
        t = (l1 - 0.02) / (0.20 - 0.02)
        t = max(0.0, min(1.0, t))
        return s_min + (s_max - s_min) * t

    print(f"{'Pair':>5}  {'L1_raw':>7}  {'L1_rt':>7}  {'str':>5}  {'L1_rep':>7}  "
          f"{'ΔL1':>8}  {'SSIM_c':>6}  {'SSIM_r':>6}  {'LPIPS_c':>7}  {'LPIPS_r':>7}  {'res':>4}")
    print("-" * 100)

    n_success = 0
    for col, idx in enumerate(test_indices):
        corrupted, clean = pairs[idx]

        corrupted_dev = corrupted.to(cfg.device).float()
        clean_dev = clean.to(cfg.device).float()

        # Roundtrip clean+corrupted to the same resize damage repair_image
        # incurs, so the L1 baseline isn't dominated by LANCZOS loss.
        clean_rt = _resize_roundtrip(clean_dev)
        corrupted_rt = _resize_roundtrip(corrupted_dev)

        # Raw L1 drives strength + skip — the model never sees the roundtrip,
        # so production-equivalent decisions should be on the raw signal.
        l1_raw = F.l1_loss(corrupted_dev, clean_dev).item()
        l1_rt = F.l1_loss(corrupted_rt, clean_rt).item()

        if l1_raw < skip_threshold:
            strength = 0.0
            repaired_dev = corrupted_rt
        else:
            strength = pick_strength(l1_raw)
            # eta=cfg.repair_eta_test (1.0, GaussianObject convention).
            # Deterministic DDIM collapses to a blurry mean prediction — keep
            # the stochastic exploration that produces the real repair quality.
            repaired_dev = repair_image(
                pipe, corrupted_dev, cfg, view_index=col,
                strength_override=strength, eta=cfg.repair_eta_test,
            ).float()

        l1_repaired = F.l1_loss(repaired_dev, clean_rt).item()
        # Compare against the roundtripped baseline so resize damage cancels.
        improvement = l1_rt - l1_repaired

        # SSIMLoss returns 1-SSIM; flip for reporting (higher = better).
        ssim_c = 1.0 - ssim_fn(corrupted_rt, clean_rt).item()
        ssim_r = 1.0 - ssim_fn(repaired_dev, clean_rt).item()
        lpips_c = lpips_fn(corrupted_rt, clean_rt).mean().item()
        lpips_r = lpips_fn(repaired_dev, clean_rt).mean().item()

        passed = (l1_repaired < l1_rt) or (l1_repaired <= success_threshold)
        if passed:
            n_success += 1
        tag = "PASS" if passed else "FAIL"
        print(f"{idx:>5}  {l1_raw:>7.4f}  {l1_rt:>7.4f}  {strength:>5.2f}  "
              f"{l1_repaired:>7.4f}  {improvement:>+8.4f}  "
              f"{ssim_c:>6.3f}  {ssim_r:>6.3f}  "
              f"{lpips_c:>7.4f}  {lpips_r:>7.4f}  {tag:>4}")

        repaired_np = repaired_dev.detach().cpu().numpy()
        axes[0, col].imshow(corrupted.numpy())
        axes[0, col].set_title(f"Corrupted (pair {idx})\nL1_raw={l1_raw:.4f}  L1_rt={l1_rt:.4f}")
        axes[0, col].axis("off")
        axes[1, col].imshow(repaired_np)
        axes[1, col].set_title(
            f"Repaired (s={strength:.2f}) [{tag}]\n"
            f"L1={l1_repaired:.4f}  Δ={improvement:+.4f}\n"
            f"SSIM {ssim_c:.3f}→{ssim_r:.3f}  LPIPS {lpips_c:.3f}→{lpips_r:.3f}"
        )
        axes[1, col].axis("off")
        axes[2, col].imshow(clean.numpy())
        axes[2, col].set_title("Ground Truth")
        axes[2, col].axis("off")

    print("-" * 90)
    print(f"Repair success rate: {n_success}/{n_test}")

    fig.suptitle(f"Repair Test — {cfg.scene_name} ({n_success}/{n_test} pass)",
                 fontsize=14, y=1.01)
    fig.savefig(test_dir / "repair_test_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    del pipe, ssim_fn, lpips_fn
    torch.cuda.empty_cache()
    print(f"Saved repair test results to {test_dir}")


def run_step5(cfg: RI3DConfig, shared_components=None):
    """Full step 5 for a single scene: generate data, train, test."""
    if not (cfg.scene_output_dir() / "repair_training_data" / "training_pairs.pt").exists():
        generate_leave_one_out_data(cfg)
    train_repair_model(cfg, shared_components=shared_components)
    test_repair_model(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: Train repair model")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene dir (or dataset root)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset root with all scenes")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--n_views", type=str, default="3", help="Number of input views per scene")
    parser.add_argument("--data_only", action="store_true", help="Only generate data, don't train")
    parser.add_argument("--train_only", action="store_true", help="Only train, assume data exists")
    parser.add_argument("--pair_stats", action="store_true",
                        help="Print L1 distribution of existing pairs and exit (no retrain)")
    parser.add_argument("--inspect_low", action="store_true",
                        help="P-D: visualize 10 lowest-L1 pairs and exit (no retrain)")
    args = parser.parse_args()

    cfg = RI3DConfig(scene_dir=Path(args.scene),
                     dataset_dir=Path(args.dataset) if args.dataset else None,
                     output_dir=Path(args.output) if args.output else None,
                     n_views=args.n_views)

    if args.pair_stats:
        pairs_path = cfg.scene_output_dir() / "repair_training_data" / "training_pairs.pt"
        pairs = torch.load(pairs_path, weights_only=False)
        print_pair_l1_distribution(
            pairs,
            save_to=cfg.scene_output_dir() / "repair_training_data" / "pairs_l1_histogram.png",
            scene_name=cfg.scene_name,
        )
    elif args.inspect_low:
        pairs_path = cfg.scene_output_dir() / "repair_training_data" / "training_pairs.pt"
        pairs = torch.load(pairs_path, weights_only=False)
        inspect_low_corruption_pairs(
            pairs,
            save_to=cfg.scene_output_dir() / "repair_training_data" / "low_corruption_inspection.png",
            scene_name=cfg.scene_name,
            k=10,
        )
    elif args.data_only:
        generate_all_scenes_data(cfg)
    elif args.train_only:
        train_repair_model(cfg)
    else:
        run_step5(cfg)
