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

    
    
    
    # Non-uniform snapshot schedule: 60% heavy-corruption + 40% light.
    # Heavy snapshots are concentrated in the first ~1300 iters after the
    # left-out view is re-added — long enough for Gaussians to densify into
    # the view's novel regions (avoiding pure black OOD backgrounds) but
    # before the floater/noise artifacts have resolved. Light snapshots
    # sample the later, near-converged tail sparsely.
    _re_add = cfg.loo_initial_iters
    # Snapshots start just after the 20-iter left-out warmup (see training
    # loop): by then Gaussians have begun densifying into the novel region
    # (see densify_stop extension above) but are still floater-heavy.
    heavy_start = _re_add + 40      # earliest: just after 35-iter warmup completes
    heavy_end = _re_add + 1000      # end of "heavy" window (fast convergence phase)
    light_start = heavy_end + 200
    light_end = cfg.loo_total_iters
    n_heavy, n_light = 12, 8        # 12 / 20 = 60%, 8 / 20 = 40%
    heavy_steps = np.linspace(heavy_start, heavy_end, n_heavy, endpoint=False).astype(int).tolist()
    light_steps = np.linspace(light_start, light_end, n_light, endpoint=False).astype(int).tolist()
    snapshot_steps = set(heavy_steps + light_steps)

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
        # Extend densification past the re-add boundary so new Gaussians can
        # form in the left-out view's novel regions during early post-re-add
        # iters. Without this, densify is off by step 6000 and those regions
        # render as black for hundreds of iters. Restored after strategy setup.
        _orig_densify_stop = cfg.densify_stop
        cfg.densify_stop = max(cfg.densify_stop, cfg.loo_initial_iters + 800)
        strategy, strategy_state = model.setup_strategy(cfg, scene_scale=loo_scene_scale)
        cfg.densify_stop = _orig_densify_stop

        w2c_lo = w2c_all[left_out_idx]
        K_lo = K_all[left_out_idx]

        # Force the first _loo_warmup_iters post-re-add iters to train on
        # the left-out view itself, so new Gaussians densify into its novel
        # regions before any snapshot is taken.
        _loo_warmup_iters = 35

        for step in tqdm(range(cfg.loo_total_iters), desc=f"  LOO {left_out_idx}"):

            if step < cfg.loo_initial_iters:
                idx = train_indices[step % len(train_indices)]
            elif step < cfg.loo_initial_iters + _loo_warmup_iters:
                idx = left_out_idx
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
    # Preview images are produced on-demand by src/extract_pairs.py.
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

            pair_corrupted_gpu.append(corrupted_r)
            pair_clean_lat_gpu.append(clean_lat.squeeze(0))

        # Augment: pre-encode horizontal flips. Doubles effective dataset size
        # at low cost (~22 MB extra latents) — important for ~40-pair training.
        # Re-encode the flipped clean image rather than flipping its latent:
        # VAE convolutions are not exactly flip-equivariant, so a flipped
        # latent doesn't match what the model would see at inference for a
        # mirrored clean image. The corrupted condition is image-space so
        # flipping it directly is exact.
        if cfg.repair_hflip_augment:
            n_orig = len(pair_corrupted_gpu)
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
    # xformers memory-efficient attention on both trained and frozen nets.
    # Crucial for fitting full ControlNet FT on a 7.6 GiB GPU. Guarded so
    # setups without xformers still run.
    try:
        import xformers  # noqa: F401
        controlnet.enable_xformers_memory_efficient_attention()
        unet.enable_xformers_memory_efficient_attention()
        print("xformers memory-efficient attention: enabled")
    except (ImportError, ModuleNotFoundError, AttributeError, Exception) as e:
        print(f"xformers not available ({type(e).__name__}); using default attention")
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
    # Min-SNR-γ weighting (Hang et al. 2023). SD 1.5 is ε-prediction, so the
    # per-sample weight is min(SNR(t), γ) / SNR(t). Precompute alphas_cumprod
    # on-device; index by timestep in the training loop.
    alphas_cumprod_dev = noise_scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)
    snr_gamma = float(cfg.repair_snr_gamma)

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
            _, h, w = c_full.shape
            x0 = random.randint(0, max(0, (w - 512) // 8)) * 8
            y0 = random.randint(0, max(0, (h - 512) // 8)) * 8
            lx, ly = x0 // 8, y0 // 8

            corrupted_b = c_full[:, y0:y0+512, x0:x0+512].unsqueeze(0).contiguous(
                memory_format=torch.channels_last)
            clean_lat = cl_full[:, ly:ly+latent_crop, lx:lx+latent_crop].unsqueeze(0)
            noise_b = torch.randn_like(clean_lat)
            t_b = torch.randint(0, max_t, (1,), device=device).long()
            noisy_lat_b = noise_scheduler.add_noise(clean_lat, noise_b, t_b).contiguous(
                memory_format=torch.channels_last)
        else:
            corrupted_crops, noisy_lats, noises, ts = [], [], [], []
            for _ in range(actual_bs):
                idx = random.randint(0, n_pairs - 1)
                c_full = pair_corrupted_gpu[idx]
                cl_full = pair_clean_lat_gpu[idx]
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
            corrupted_b = torch.cat(corrupted_crops).contiguous(memory_format=torch.channels_last)
            noisy_lat_b = torch.cat(noisy_lats).contiguous(memory_format=torch.channels_last)
            noise_b = torch.cat(noises)
            t_b = torch.cat(ts)

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

        # Min-SNR-γ weighted ε-prediction loss. Per-sample weight
        # w(t) = min(SNR(t), γ) / SNR(t) rebalances mid/low-t (structure)
        # against high-t (pure noise) samples for faster convergence.
        ac = alphas_cumprod_dev[t_b]                       # [B]
        snr = ac / (1.0 - ac).clamp_min(1e-8)              # [B]
        w = torch.clamp(snr, max=snr_gamma) / snr          # [B]
        per_sample_mse = (noise_pred.float() - noise_b.float()).pow(2).mean(dim=[1, 2, 3])
        loss = (w * per_sample_mse).mean()
        loss.backward()
        # Speed: clip but DO NOT call .isfinite() — the boolean check forces a
        # GPU↔CPU sync every step. Warmup keeps loss stable early; if NaN
        # ever appears, optimizer.step on NaN grads only affects that one step.
        torch.nn.utils.clip_grad_norm_(controlnet.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()

        iter_count += actual_bs
        # Log the UNWEIGHTED per-sample MSE — it's the actual ε-prediction
        # quality and stable across timesteps. The Min-SNR-γ w(t) swings ~200×
        # per step just from the random t draw, so plotting the weighted loss
        # hides convergence under sampling noise. Keep w(t) for backward only.
        losses.append(per_sample_mse.mean().detach())
        if iter_count % 100 < batch_size or step == n_steps - 1:
            recent_t = torch.stack(losses[-max(1, 100 // batch_size):]).float()
            avg_loss = recent_t.mean().item()
            print(f"  Step ~{min(iter_count, n_iters)}: mse = {avg_loss:.6f}")

    controlnet.save_pretrained(model_dir / "controlnet")
    print(f"Saved repair model (full ControlNet) to {model_dir}")

    losses_cpu = torch.stack(losses).float().cpu().numpy() if losses else np.array([])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses_cpu, alpha=0.3, label="per-step MSE")
    if len(losses_cpu) >= 10:
        win = max(10, len(losses_cpu) // 50)
        kernel = np.ones(win) / win
        rolling = np.convolve(losses_cpu, kernel, mode="valid")
        ax.plot(np.arange(win - 1, len(losses_cpu)), rolling,
                color="C1", linewidth=2, label=f"rolling mean (w={win})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Unweighted ε-MSE")
    ax.set_title(f"Repair Model Training Loss — {cfg.scene_name}")
    ax.legend()
    fig.savefig(model_dir / "training_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    del controlnet, optimizer, pair_corrupted_gpu, pair_clean_lat_gpu, losses
    if _owns_components:
        del unet
    torch.cuda.empty_cache()


def _prepare_for_pipeline(image: Image.Image, target_short_side: int = 512) -> tuple[Image.Image, int, int]:
    """Deprecated: use utils.prepare_for_pipeline instead. Kept for import compat."""
    from utils import prepare_for_pipeline
    return prepare_for_pipeline(image, target_short_side)


def test_repair_model(cfg: RI3DConfig):
    """Test the trained repair model using the same pipeline as Stage 1.

    Uses adaptive strength based on initial perceptual corruption severity
    (LPIPS). Reports PASS/FAIL per pair: success if LPIPS_repaired <
    LPIPS_corrupted OR LPIPS_repaired <= cfg.repair_test_lpips_success.
    LPIPS (lower = better) reflects perceptual repair quality; L1 can be
    gamed by blur regressing to the mean.
    """
    from step6_stage1_optim import load_repair_pipeline, repair_image
    from gaussian_trainer import LPIPSLoss

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
    lpips_fn = LPIPSLoss(device=cfg.device)

    # Rank pairs by LPIPS and test across the perceptual spectrum, biased
    # toward the heavy-corruption tail (the regime that's failing).
    with torch.no_grad():
        all_lpips = []
        for corrupted, clean in pairs:
            all_lpips.append(lpips_fn(corrupted.to(cfg.device), clean.to(cfg.device)).item())
    order = sorted(range(len(pairs)), key=lambda i: all_lpips[i])
    n_test = min(6, len(pairs))
    # Quantile picks across sorted order → always includes easiest, hardest.
    test_indices = [order[int(i * (len(order) - 1) / max(n_test - 1, 1))]
                    for i in range(n_test)]

    fig, axes = plt.subplots(3, n_test, figsize=(5 * n_test, 15))
    if n_test == 1:
        axes = axes.reshape(-1, 1)

    s_max = cfg.repair_strength_max
    s_min = cfg.repair_strength_min
    success_threshold = cfg.repair_test_lpips_success

    def pick_strength(lp: float) -> float:
        # Map LPIPS ∈ [0.05, 0.40] → strength ∈ [s_min, s_max]; clamp outside.
        t = (lp - 0.05) / (0.40 - 0.05)
        t = max(0.0, min(1.0, t))
        return s_min + (s_max - s_min) * t

    print(f"{'Pair':>6}  {'LP_corr':>8}  {'strength':>8}  {'LP_rep':>7}  "
          f"{'Δ':>8}  {'result':>6}")
    print("-" * 60)

    n_success = 0
    per_pair = []
    for col, idx in enumerate(test_indices):
        corrupted, clean = pairs[idx]

        corrupted_dev = corrupted.to(cfg.device)
        clean_dev = clean.to(cfg.device)
        lp_corrupted = lpips_fn(corrupted_dev, clean_dev).item()

        strength = pick_strength(lp_corrupted)
        repaired_dev = repair_image(pipe, corrupted_dev, cfg, view_index=col,
                                     strength_override=strength,
                                     eta=cfg.repair_eta_test)
        lp_repaired = lpips_fn(repaired_dev, clean_dev).item()
        improvement = lp_corrupted - lp_repaired

        passed = (lp_repaired < lp_corrupted) or (lp_repaired <= success_threshold)
        if passed:
            n_success += 1
        tag = "PASS" if passed else "FAIL"
        print(f"{idx:>6}  {lp_corrupted:>8.4f}  {strength:>8.2f}  {lp_repaired:>7.4f}  "
              f"{improvement:>+8.4f}  {tag:>6}")
        per_pair.append((idx, lp_corrupted, lp_repaired))

        repaired_np = repaired_dev.detach().cpu().numpy()
        axes[0, col].imshow(corrupted.numpy())
        axes[0, col].set_title(f"Corrupted (pair {idx})\nLPIPS={lp_corrupted:.4f}")
        axes[0, col].axis("off")
        axes[1, col].imshow(repaired_np)
        axes[1, col].set_title(
            f"Repaired (s={strength:.2f}) [{tag}]\n"
            f"LPIPS={lp_repaired:.4f}  Δ={improvement:+.4f}"
        )
        axes[1, col].axis("off")
        axes[2, col].imshow(clean.numpy())
        axes[2, col].set_title("Ground Truth")
        axes[2, col].axis("off")

    mean_lp_rep = sum(p[2] for p in per_pair) / max(len(per_pair), 1)
    print("-" * 60)
    print(f"Repair success rate: {n_success}/{n_test}  |  mean LPIPS_rep = {mean_lp_rep:.4f}")
    torch.save({"per_pair": per_pair, "mean_lpips_repaired": mean_lp_rep,
                "n_success": n_success, "n_test": n_test},
               test_dir / "test_metrics.pt")

    fig.suptitle(f"Repair Test — {cfg.scene_name} ({n_success}/{n_test} pass, LPIPS)",
                 fontsize=14, y=1.01)
    fig.savefig(test_dir / "repair_test_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    del pipe, lpips_fn
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
    args = parser.parse_args()

    cfg = RI3DConfig(scene_dir=Path(args.scene),
                     dataset_dir=Path(args.dataset) if args.dataset else None,
                     output_dir=Path(args.output) if args.output else None,
                     n_views=args.n_views)

    if args.data_only:
        generate_all_scenes_data(cfg)
    elif args.train_only:
        train_repair_model(cfg)
    else:
        run_step5(cfg)
