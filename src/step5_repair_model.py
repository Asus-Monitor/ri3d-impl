"""Step 5: Train the Repair Model (ControlNet + LoRA).

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

    n_pairs = len(pair_corrupted_gpu)
    print(f"Pre-encoded {n_pairs} pairs on GPU")

    
    if _owns_components:
        del vae
    torch.cuda.empty_cache()

    
    controlnet = ControlNetModel.from_pretrained(cfg.controlnet_model,
                                                  torch_dtype=dtype, use_safetensors=False).to(device)
    lora_alpha = int(cfg.repair_lora_rank * cfg.repair_lora_alpha_mult)
    lora_config = LoraConfig(
        r=cfg.repair_lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "to_q", "to_v", "to_k", "to_out.0",  
            "conv1", "conv2",                       
            "proj_in", "proj_out",                  
            "controlnet_cond_embedding.conv_in",
            "controlnet_cond_embedding.blocks.0",
            "controlnet_cond_embedding.blocks.1",
            "controlnet_cond_embedding.blocks.2",
            "controlnet_cond_embedding.blocks.3",
            "controlnet_cond_embedding.blocks.4",
            "controlnet_cond_embedding.blocks.5",
            "controlnet_cond_embedding.conv_out",
        ],
        lora_dropout=cfg.repair_lora_dropout,
    )
    controlnet = get_peft_model(controlnet, lora_config)
    controlnet.print_trainable_parameters()

    
    unet.requires_grad_(False)
    unet.eval()

    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=cfg.repair_lr, weight_decay=1e-2)

    n_iters = cfg.repair_train_iters
    max_t = noise_scheduler.config.num_train_timesteps

    
    gpu_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    if gpu_mem_gb >= 20:
        batch_size = 4
    elif gpu_mem_gb >= 7:
        batch_size = 2
    else:
        batch_size = 1
    n_steps = (n_iters + batch_size - 1) // batch_size
    text_embeds_b = text_embeds.expand(batch_size, -1, -1)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=cfg.repair_lr * 0.01,
    )

    scaler = torch.amp.GradScaler("cuda")

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

            
            x0 = random.randint(0, max(0, (w - 512) // 8)) * 8
            y0 = random.randint(0, max(0, (h - 512) // 8)) * 8
            lx, ly = x0 // 8, y0 // 8

            corrupted_crops.append(c_full[:, y0:y0+512, x0:x0+512].unsqueeze(0))
            clean_lat = cl_full[:, ly:ly+latent_crop, lx:lx+latent_crop].unsqueeze(0)

            noise = torch.randn_like(clean_lat)
            t = torch.randint(0, max_t, (1,), device=device).long()

            noisy_lats.append(noise_scheduler.add_noise(clean_lat, noise, t))
            noises.append(noise)
            ts.append(t)

        corrupted_b = torch.cat(corrupted_crops)
        noisy_lat_b = torch.cat(noisy_lats)
        noise_b = torch.cat(noises)
        t_b = torch.cat(ts)

        step_embeds = text_embeds_b[:actual_bs]

        with torch.amp.autocast("cuda"):
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

        
        loss = F.mse_loss(noise_pred.float(), noise_b.float())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(controlnet.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()

        iter_count += actual_bs
        losses.append(loss.item())
        if iter_count % 100 < batch_size or step == n_steps - 1:
            recent = losses[-max(1, 100 // batch_size):]
            avg_loss = sum(recent) / len(recent)
            print(f"  Step ~{min(iter_count, n_iters)}: loss = {avg_loss:.6f}")

    controlnet.save_pretrained(model_dir / "controlnet")
    (model_dir / "unet_frozen").touch()
    print(f"Saved repair model (ControlNet LoRA incl. cond_embedding) to {model_dir}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Repair Model Training Loss — {cfg.scene_name}")
    fig.savefig(model_dir / "training_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    del controlnet, optimizer, pair_corrupted_gpu, pair_clean_lat_gpu
    if _owns_components:
        del unet
    torch.cuda.empty_cache()


def _prepare_for_pipeline(image: Image.Image, target_short_side: int = 512) -> tuple[Image.Image, int, int]:
    """Deprecated: use utils.prepare_for_pipeline instead. Kept for import compat."""
    from utils import prepare_for_pipeline
    return prepare_for_pipeline(image, target_short_side)


def test_repair_model(cfg: RI3DConfig):
    """Test the trained repair model using the same pipeline and inference
    path as the actual optimization (load_repair_pipeline + repair_image)."""
    from step6_stage1_optim import load_repair_pipeline, repair_image

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

    n_test = min(4, len(pairs))
    fig, axes = plt.subplots(3, n_test, figsize=(5 * n_test, 15))
    if n_test == 1:
        axes = axes.reshape(-1, 1)

    print(f"{'Pair':>6}  {'Corrupted L1':>13}  {'Repaired L1':>12}  {'Improvement':>11}")
    print("-" * 50)

    for j in range(n_test):
        idx = j * (len(pairs) // n_test)
        corrupted, clean = pairs[idx]

        
        corrupted_tensor = corrupted.to(cfg.device)
        repaired_tensor = repair_image(pipe, corrupted_tensor, cfg, view_index=j)

        
        clean_device = clean.to(cfg.device)
        l1_corrupted = F.l1_loss(corrupted_tensor, clean_device).item()
        l1_repaired = F.l1_loss(repaired_tensor, clean_device).item()
        improvement = l1_corrupted - l1_repaired
        print(f"{idx:>6}  {l1_corrupted:>13.4f}  {l1_repaired:>12.4f}  {improvement:>+11.4f}")

        result_np = repaired_tensor.cpu().numpy()

        axes[0, j].imshow(corrupted.numpy())
        axes[0, j].set_title(f"Corrupted (pair {idx})\nL1={l1_corrupted:.4f}")
        axes[0, j].axis("off")
        axes[1, j].imshow(result_np)
        axes[1, j].set_title(f"Repaired\nL1={l1_repaired:.4f} ({improvement:+.4f})")
        axes[1, j].axis("off")
        axes[2, j].imshow(clean.numpy())
        axes[2, j].set_title("Ground Truth")
        axes[2, j].axis("off")

    fig.suptitle(f"Repair Test — {cfg.scene_name}", fontsize=14, y=1.01)
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
