"""Step 5: Train the Repair Model (ControlNet + LoRA + LCM).

Per the paper (Sec 4.2, 8.1):
  1. Leave-one-out: create N subsets of N-1 images, train N 3DGS reps
     (6000 iters without left-out view, then to 10000 with it re-added)
  2. Snapshot intermediate renders -> training pairs (corrupted, clean)
  3. Fine-tune ControlNet (tile) with LoRA on this scene's pairs for 1800 iters

Models are personalized per-scene so the repair model matches the scene's visual style.

Outputs:
  - outputs/<scene>/repair_training_data/  corrupted/clean pairs
  - outputs/<scene>/repair_model/          fine-tuned ControlNet LoRA weights
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

    # Load GT at DUSt3R resolution (for 3DGS training and repair pairs)
    gt_images_render = []
    for ip in image_paths:
        img = Image.open(ip).convert("RGB")
        gt_images_render.append(
            torch.from_numpy(np.array(img.resize((W, H), Image.LANCZOS))).float() / 255.0
        )

    all_pairs = []

    # Pre-compute w2c matrices and move everything to GPU once
    w2c_all = torch.linalg.inv(poses).to(device)
    K_all = intrinsics.to(device)

    # Downscaled resolution for LOO training — rasterization cost scales with pixel count.
    # Snapshots still render at full res for quality pairs.
    s = cfg.loo_render_scale
    loo_H, loo_W = int(H * s), int(W * s)
    loo_K_all = K_all.clone()
    loo_K_all[:, 0, :] *= s  # scale fx, cx
    loo_K_all[:, 1, :] *= s  # scale fy, cy

    # GT images at LOO resolution for training loss (on GPU)
    gt_gpu_loo = []
    for g in gt_images_render:
        g_loo = F.interpolate(
            g.permute(2, 0, 1).unsqueeze(0), size=(loo_H, loo_W),
            mode="bilinear", align_corners=False,
        ).squeeze(0).permute(1, 2, 0).to(device)
        gt_gpu_loo.append(g_loo)

    torch.backends.cudnn.benchmark = True

    # Snapshot only during phase 2 (after re-adding the left-out view), per paper Sec 4.2.
    # Phase 1 renders are maximally corrupted (model never saw this view) — including
    # them poisons LoRA training with impossible extreme-corruption→clean mappings.
    # Phase 2 gives progressive corruption: moderate→mild, matching what the repair
    # model actually encounters during Stage 1 (heavy corruption is masked out by M_α).
    snapshot_steps = set(
        range(cfg.loo_initial_iters, cfg.loo_total_iters, cfg.loo_snapshot_interval)
    )

    all_indices = list(range(n_images))

    for left_out_idx in range(n_images):
        name = Path(image_paths[left_out_idx]).stem
        print(f"\nLeave-one-out: excluding view {left_out_idx} ({name})")

        train_indices = [j for j in range(n_images) if j != left_out_idx]
        clean_cpu = gt_images_render[left_out_idx]

        model = GaussianModel(gaussians_init, device)
        optimizers = model.setup_optimizers(cfg)
        # Compute scene scale from camera positions (same as Stage 1/2)
        cam_positions = poses[:, :3, 3]
        loo_scene_scale = max(
            (cam_positions - cam_positions.mean(dim=0)).norm(dim=1).mean().item(), 0.1
        )
        strategy, strategy_state = model.setup_strategy(cfg, scene_scale=loo_scene_scale)

        w2c_lo = w2c_all[left_out_idx]
        K_lo = K_all[left_out_idx]

        for step in tqdm(range(cfg.loo_total_iters), desc=f"  LOO {left_out_idx}"):
            # Phase 1: train without left-out view; Phase 2: all views
            if step < cfg.loo_initial_iters:
                idx = train_indices[step % len(train_indices)]
            else:
                idx = all_indices[step % n_images]

            # Train at reduced resolution (L1-only — SSIM conv2d overhead not needed
            # for generating corruption patterns, paper doesn't specify LOO loss)
            result = model.render_for_optim(
                w2c_all[idx], loo_K_all[idx], loo_H, loo_W,
                strategy, strategy_state, step,
                render_mode="RGB",
            )
            loss = F.l1_loss(result["image"], gt_gpu_loo[idx])
            loss.backward()
            model.step_post_backward(step, result["meta"])
            model.optimizer_step()

            # Snapshot at key points during both phases
            # Full resolution — these become the actual repair training pairs
            if step in snapshot_steps:
                with torch.no_grad():
                    r = model.render(w2c_lo, K_lo, H, W)
                    corrupted = r["image"].clamp(0, 1).cpu()
                    all_pairs.append((corrupted, clean_cpu))

        del model
        torch.cuda.empty_cache()

    print(f"\nGenerated {len(all_pairs)} training pairs for scene {cfg.scene_name}")
    torch.save(all_pairs, data_dir / "training_pairs.pt")

    # Save visual preview — sample evenly across all pairs to show different views
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
        # Check if steps 1-4 outputs exist
        if not (scene_cfg.scene_output_dir() / "init_gaussians.pt").exists():
            print(f"  Skipping {scene_dir.name}: run steps 1-4 first")
            continue
        # Check if already generated
        if (scene_cfg.scene_output_dir() / "repair_training_data" / "training_pairs.pt").exists():
            print(f"  Skipping {scene_dir.name}: training pairs already exist")
            continue
        generate_leave_one_out_data(scene_cfg)


def train_repair_model(cfg: RI3DConfig, shared_components=None):
    """Train a per-scene repair model (ControlNet LoRA) on this scene's leave-one-out data.

    Per the paper: fine-tune the ControlNet on corrupted/clean pairs so it learns
    scene-specific repair. The UNet stays frozen.

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
    from peft import LoraConfig, get_peft_model

    model_dir = cfg.scene_output_dir() / "repair_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device
    dtype = cfg.dtype

    # Load this scene's training pairs
    pairs_path = cfg.scene_output_dir() / "repair_training_data" / "training_pairs.pt"
    all_pairs = torch.load(pairs_path, weights_only=False)
    if len(all_pairs) == 0:
        raise RuntimeError(f"No training pairs for {cfg.scene_name}. Run data generation first.")
    print(f"Training pairs for {cfg.scene_name}: {len(all_pairs)}")

    # Use shared frozen components if provided, otherwise load our own
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

    # Encode the SAME positive prompt used at inference (cfg.repair_positive_prompt).
    # Training with empty "" but inferring with "best quality, sharp detail" creates
    # a text-embedding mismatch that distorts the LoRA's learned corrections.
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
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Pre-compute: resize pairs, move to GPU, pre-encode clean latents.
    # Eliminates per-step VAE encode + CPU resize + CPU→GPU transfer.
    # ------------------------------------------------------------------
    torch.backends.cudnn.benchmark = True
    latent_crop = 64  # 512px // 8 (VAE downscale)
    pair_corrupted_gpu = []   # (3, H, W) on GPU, dims rounded to 8
    pair_clean_lat_gpu = []   # (4, H//8, W//8) pre-encoded latents

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

    n_pairs = len(pair_corrupted_gpu)
    print(f"Pre-encoded {n_pairs} pairs on GPU")

    # Fresh ControlNet + LoRA for this scene (UNet stays frozen)
    controlnet = ControlNetModel.from_pretrained(cfg.controlnet_model,
                                                  torch_dtype=dtype, use_safetensors=False).to(device)
    lora_config = LoraConfig(
        r=cfg.repair_lora_rank,
        lora_alpha=cfg.repair_lora_rank,
        target_modules=[
            "to_q", "to_v", "to_k", "to_out.0",  # attention
            "conv1", "conv2",                       # resnet convolutions
            "proj_in", "proj_out",                  # transformer projections
        ],
        lora_dropout=0.0,
    )
    controlnet = get_peft_model(controlnet, lora_config)
    controlnet.print_trainable_parameters()

    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=cfg.repair_lr, weight_decay=1e-2)

    n_iters = cfg.repair_train_iters
    max_t = noise_scheduler.config.num_train_timesteps

    # Scale batch size to GPU memory
    gpu_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    if gpu_mem_gb >= 20:
        batch_size = 4
    elif gpu_mem_gb >= 12:
        batch_size = 2
    else:
        batch_size = 1
    n_steps = (n_iters + batch_size - 1) // batch_size
    text_embeds_b = text_embeds.expand(batch_size, -1, -1)

    print(f"Training ControlNet LoRA for {n_iters} iterations (batch={batch_size}) "
          f"on {n_pairs} pairs...")
    losses = []
    iter_count = 0

    for step in tqdm(range(n_steps), desc=f"Repair [{cfg.scene_name}]"):
        actual_bs = min(batch_size, n_iters - iter_count)
        if actual_bs <= 0:
            break

        corrupted_crops = []
        noisy_lats = []
        noises = []
        ts = []

        for _ in range(actual_bs):
            idx = random.randint(0, n_pairs - 1)
            c_full = pair_corrupted_gpu[idx]
            cl_full = pair_clean_lat_gpu[idx]
            _, h, w = c_full.shape

            # Random crop aligned to 8px for exact latent correspondence
            x0 = random.randint(0, max(0, (w - 512) // 8)) * 8
            y0 = random.randint(0, max(0, (h - 512) // 8)) * 8
            lx, ly = x0 // 8, y0 // 8

            corrupted_crops.append(c_full[:, y0:y0+512, x0:x0+512].unsqueeze(0))
            clean_lat = cl_full[:, ly:ly+latent_crop, lx:lx+latent_crop].unsqueeze(0)

            noise = torch.randn_like(clean_lat)
            t = torch.randint(0, max_t, (1,), device=device).long()

            # Standard training per paper: noise clean latent, predict epsilon.
            # ControlNet condition (corrupted image) teaches the mapping.
            noisy_lats.append(noise_scheduler.add_noise(clean_lat, noise, t))
            noises.append(noise)
            ts.append(t)

        corrupted_b = torch.cat(corrupted_crops)
        noisy_lat_b = torch.cat(noisy_lats)
        noise_b = torch.cat(noises)
        t_b = torch.cat(ts)

        controlnet_output = controlnet(
            noisy_lat_b, t_b, encoder_hidden_states=text_embeds_b[:actual_bs],
            controlnet_cond=corrupted_b,
            return_dict=False,
        )

        noise_pred = unet(
            noisy_lat_b, t_b, encoder_hidden_states=text_embeds_b[:actual_bs],
            down_block_additional_residuals=controlnet_output[0],
            mid_block_additional_residual=controlnet_output[1],
        ).sample

        loss = F.mse_loss(noise_pred.float(), noise_b.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        iter_count += actual_bs
        losses.append(loss.item())
        if iter_count % 100 < batch_size or step == n_steps - 1:
            recent = losses[-max(1, 100 // batch_size):]
            avg_loss = sum(recent) / len(recent)
            print(f"  Step ~{min(iter_count, n_iters)}: loss = {avg_loss:.6f}")

    controlnet.save_pretrained(model_dir)
    print(f"Saved repair model to {model_dir}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Repair Model Training Loss — {cfg.scene_name}")
    fig.savefig(model_dir / "training_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    del controlnet, optimizer, pair_corrupted_gpu, pair_clean_lat_gpu
    if _owns_components:
        del text_encoder, vae, unet
    torch.cuda.empty_cache()


def _prepare_for_pipeline(image: Image.Image, target_short_side: int = 512) -> tuple[Image.Image, int, int]:
    """Resize preserving aspect ratio for pipeline input.

    Matches the training preprocessing: smallest dim → target_short_side,
    then dims rounded to multiples of 8 for the VAE.
    Returns (resized_pil, pipe_h, pipe_w).
    """
    W_orig, H_orig = image.size
    if H_orig <= W_orig:
        pipe_h = target_short_side
        pipe_w = int(W_orig * target_short_side / H_orig)
    else:
        pipe_w = target_short_side
        pipe_h = int(H_orig * target_short_side / W_orig)
    # Round to multiples of 8 (VAE latent alignment)
    pipe_h = (pipe_h // 8) * 8
    pipe_w = (pipe_w // 8) * 8
    resized = image.resize((pipe_w, pipe_h), Image.LANCZOS)
    return resized, pipe_h, pipe_w


def test_repair_model(cfg: RI3DConfig):
    """Test the trained repair model on corrupted images from this scene."""
    from diffusers import (
        StableDiffusionControlNetImg2ImgPipeline,
        ControlNetModel,
        DPMSolverMultistepScheduler,
    )
    from peft import PeftModel

    out_dir = cfg.scene_output_dir()
    test_dir = out_dir / "repair_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device
    dtype = cfg.dtype

    pairs_path = out_dir / "repair_training_data" / "training_pairs.pt"
    if not pairs_path.exists():
        print("No training pairs for this scene, skipping test.")
        return
    pairs = torch.load(pairs_path, weights_only=False)

    model_dir = cfg.scene_output_dir() / "repair_model"
    print("Loading repair pipeline for test...")
    controlnet = ControlNetModel.from_pretrained(cfg.controlnet_model, torch_dtype=dtype, use_safetensors=False)
    controlnet = PeftModel.from_pretrained(controlnet, model_dir)
    controlnet = controlnet.merge_and_unload()

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        cfg.sd_model, controlnet=controlnet, torch_dtype=dtype,
    ).to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True)
    pipe.safety_checker = None

    n_test = min(4, len(pairs))
    fig, axes = plt.subplots(3, n_test, figsize=(5 * n_test, 15))
    if n_test == 1:
        axes = axes.reshape(-1, 1)

    for j in range(n_test):
        idx = j * (len(pairs) // n_test)
        corrupted, clean = pairs[idx]

        corrupted_pil = Image.fromarray(
            (corrupted.numpy() * 255).astype(np.uint8)
        )
        corrupted_resized, pipe_h, pipe_w = _prepare_for_pipeline(corrupted_pil)

        with torch.no_grad():
            result = pipe(
                prompt=cfg.repair_positive_prompt,
                negative_prompt=cfg.repair_negative_prompt,
                image=corrupted_resized,
                control_image=corrupted_resized,
                strength=cfg.repair_strength,
                num_inference_steps=cfg.repair_inference_steps,
                guidance_scale=cfg.repair_guidance_scale,
                controlnet_conditioning_scale=cfg.repair_controlnet_scale,
            ).images[0]

        # Resize result back to original pair resolution for comparison
        H_orig, W_orig = corrupted.shape[:2]
        result = result.resize((W_orig, H_orig), Image.LANCZOS)

        axes[0, j].imshow(corrupted.numpy())
        axes[0, j].set_title("Corrupted")
        axes[0, j].axis("off")
        axes[1, j].imshow(result)
        axes[1, j].set_title("Repaired")
        axes[1, j].axis("off")
        axes[2, j].imshow(clean.numpy())
        axes[2, j].set_title("Ground Truth")
        axes[2, j].axis("off")

    fig.suptitle(f"Repair Test — {cfg.scene_name}")
    fig.savefig(test_dir / "repair_test_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    del pipe
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
