"""Step 5: Train the Repair Model (ControlNet + LoRA + LCM).

Per the paper (Sec 4.2, 8.1):
  1. Leave-one-out: create N subsets of N-1 images
  2. Train N small 3DGS representations (6000 iters)
  3. Re-add left-out image, continue to 10000 iters
  4. Snapshot intermediate renders at various stages -> training pairs (corrupted, clean)
  5. Fine-tune ControlNet (tile) with LoRA on these pairs for 1800 iters

Outputs:
  - outputs/<scene>/repair_model/          fine-tuned ControlNet LoRA weights
  - outputs/<scene>/repair_training_data/  corrupted/clean pairs (for inspection)
  - outputs/<scene>/repair_test/           test repair results
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
    GaussianModel, SSIMLoss, reconstruction_loss, PlateauDetector
)


def generate_leave_one_out_data(cfg: RI3DConfig):
    """Generate corrupted/clean image pairs using leave-one-out 3DGS training."""
    from gsplat import rasterization, DefaultStrategy

    out_dir = cfg.scene_output_dir()
    data_dir = out_dir / "repair_training_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load scene data
    image_paths = torch.load(out_dir / "image_paths.pt", weights_only=False)
    poses = torch.load(out_dir / "dust3r_poses.pt", weights_only=True).float()
    intrinsics = torch.load(out_dir / "dust3r_intrinsics.pt", weights_only=True).float()
    gaussians_init = torch.load(out_dir / "init_gaussians.pt", weights_only=True)
    n_images = len(image_paths)

    device = cfg.device
    fused_depth = torch.load(out_dir / "fused_depths" / "fused_depth_000.pt", weights_only=True)
    H, W = fused_depth.shape

    # Load GT images at render resolution
    gt_images = []
    for ip in image_paths:
        img = Image.open(ip).convert("RGB")
        img = img.resize((W, H), Image.LANCZOS)
        gt_images.append(torch.from_numpy(np.array(img)).float() / 255.0)

    ssim_fn = SSIMLoss().to(device)
    all_pairs = []  # list of (corrupted_tensor, clean_tensor)

    for left_out_idx in range(n_images):
        name = Path(image_paths[left_out_idx]).stem
        print(f"\nLeave-one-out: excluding view {left_out_idx} ({name})")

        # Training views = all except left_out_idx
        train_indices = [j for j in range(n_images) if j != left_out_idx]
        clean_image = gt_images[left_out_idx].to(device)

        # Initialize Gaussians (fresh copy)
        model = GaussianModel(gaussians_init, device)
        optimizers = model.setup_optimizers(cfg)
        strategy, strategy_state = model.setup_strategy(cfg, scene_scale=1.0)

        # Phase 1: Train without left-out view (6000 iters)
        print(f"  Phase 1: training on {len(train_indices)} views for {cfg.loo_initial_iters} iters...")
        for step in range(cfg.loo_initial_iters):
            idx = train_indices[step % len(train_indices)]
            w2c = torch.linalg.inv(poses[idx]).to(device)
            K = intrinsics[idx].to(device)
            gt = gt_images[idx].to(device)

            result = model.render_for_optim(w2c, K, H, W, strategy, strategy_state, step)
            loss = reconstruction_loss(result["image"], gt, ssim_fn, cfg=cfg)
            loss.backward()
            model.step_post_backward(step, result["meta"])
            model.optimizer_step()

        # Snapshot: render left-out view as corrupted
        with torch.no_grad():
            w2c_lo = torch.linalg.inv(poses[left_out_idx]).to(device)
            K_lo = intrinsics[left_out_idx].to(device)
            r = model.render(w2c_lo, K_lo, H, W)
            corrupted_early = r["image"].clamp(0, 1).cpu()
            all_pairs.append((corrupted_early, gt_images[left_out_idx]))

        # Phase 2: Re-add left-out view, continue training
        print(f"  Phase 2: re-adding view, training to {cfg.loo_total_iters} iters...")
        all_indices = list(range(n_images))
        for step in range(cfg.loo_initial_iters, cfg.loo_total_iters):
            idx = all_indices[step % n_images]
            w2c = torch.linalg.inv(poses[idx]).to(device)
            K = intrinsics[idx].to(device)
            gt = gt_images[idx].to(device)

            result = model.render_for_optim(w2c, K, H, W, strategy, strategy_state, step)
            loss = reconstruction_loss(result["image"], gt, ssim_fn, cfg=cfg)
            loss.backward()
            model.step_post_backward(step, result["meta"])
            model.optimizer_step()

            # Snapshot intermediate renders of left-out view
            if (step - cfg.loo_initial_iters) % cfg.loo_snapshot_interval == 0:
                with torch.no_grad():
                    r = model.render(w2c_lo, K_lo, H, W)
                    corrupted = r["image"].clamp(0, 1).cpu()
                    all_pairs.append((corrupted, gt_images[left_out_idx]))

        # Final snapshot
        with torch.no_grad():
            r = model.render(w2c_lo, K_lo, H, W)
            corrupted_final = r["image"].clamp(0, 1).cpu()
            all_pairs.append((corrupted_final, gt_images[left_out_idx]))

        del model
        torch.cuda.empty_cache()

    # Save training pairs
    print(f"\nGenerated {len(all_pairs)} training pairs")
    torch.save(all_pairs, data_dir / "training_pairs.pt")

    # Save visual examples
    n_show = min(8, len(all_pairs))
    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
    for j in range(n_show):
        axes[0, j].imshow(all_pairs[j][0].numpy())
        axes[0, j].set_title(f"Corrupted {j}")
        axes[0, j].axis("off")
        axes[1, j].imshow(all_pairs[j][1].numpy())
        axes[1, j].set_title(f"Clean {j}")
        axes[1, j].axis("off")
    fig.suptitle("Leave-One-Out Training Pairs")
    fig.savefig(data_dir / "pairs_preview.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved training data to {data_dir}")
    return all_pairs


def train_repair_model(cfg: RI3DConfig):
    """Fine-tune ControlNet (tile) with LoRA on corrupted/clean pairs."""
    from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        DDPMScheduler,
        AutoencoderKL,
        UNet2DConditionModel,
    )
    from diffusers.utils import load_image
    from transformers import CLIPTextModel, CLIPTokenizer
    from peft import LoraConfig, get_peft_model

    out_dir = cfg.scene_output_dir()
    model_dir = out_dir / "repair_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device
    dtype = cfg.dtype

    # Load training pairs
    pairs = torch.load(out_dir / "repair_training_data" / "training_pairs.pt", weights_only=False)
    print(f"Loaded {len(pairs)} training pairs")

    # Load models
    print("Loading SD 1.5 + ControlNet tile...")
    tokenizer = CLIPTokenizer.from_pretrained(cfg.sd_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.sd_model, subfolder="text_encoder",
                                                  torch_dtype=dtype).to(device)
    vae = AutoencoderKL.from_pretrained(cfg.sd_model, subfolder="vae",
                                         torch_dtype=dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(cfg.sd_model, subfolder="unet",
                                                 torch_dtype=dtype).to(device)
    controlnet = ControlNetModel.from_pretrained(cfg.controlnet_model,
                                                  torch_dtype=dtype).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.sd_model, subfolder="scheduler")

    # Freeze everything except ControlNet
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # Add LoRA to ControlNet for efficient fine-tuning
    lora_config = LoraConfig(
        r=cfg.repair_lora_rank,
        lora_alpha=cfg.repair_lora_rank,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"],
        lora_dropout=0.0,
    )
    controlnet = get_peft_model(controlnet, lora_config)
    controlnet.print_trainable_parameters()

    # Optimizer
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=cfg.repair_lr, weight_decay=1e-2)

    # Prepare empty text embedding (unconditional, scene-agnostic)
    with torch.no_grad():
        tokens = tokenizer("", padding="max_length", max_length=77,
                           return_tensors="pt").input_ids.to(device)
        text_embeds = text_encoder(tokens)[0]  # (1, 77, 768)

    print(f"Training ControlNet LoRA for {cfg.repair_train_iters} iterations...")
    losses = []

    for step in tqdm(range(cfg.repair_train_iters), desc="Repair training"):
        # Sample a random pair
        corrupted, clean = random.choice(pairs)

        # Resize to 512x512 for diffusion
        corrupted_512 = F.interpolate(
            corrupted.permute(2, 0, 1).unsqueeze(0), size=(512, 512),
            mode="bilinear", align_corners=False
        ).to(device, dtype)  # (1, 3, 512, 512)

        clean_512 = F.interpolate(
            clean.permute(2, 0, 1).unsqueeze(0), size=(512, 512),
            mode="bilinear", align_corners=False
        ).to(device, dtype)  # (1, 3, 512, 512)

        # Encode clean image to latent
        with torch.no_grad():
            clean_latent = vae.encode(clean_512 * 2 - 1).latent_dist.sample() * vae.config.scaling_factor

        # Sample noise and timestep
        noise = torch.randn_like(clean_latent)
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,),
                          device=device).long()

        # Add noise to clean latent
        noisy_latent = noise_scheduler.add_noise(clean_latent, noise, t)

        # ControlNet forward: condition on corrupted image
        controlnet_output = controlnet(
            noisy_latent, t, encoder_hidden_states=text_embeds,
            controlnet_cond=corrupted_512,
            return_dict=False,
        )
        down_block_res_samples = controlnet_output[0]
        mid_block_res_sample = controlnet_output[1]

        # UNet prediction with ControlNet residuals
        noise_pred = unet(
            noisy_latent, t, encoder_hidden_states=text_embeds,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample

        # MSE loss on noise prediction
        loss = F.mse_loss(noise_pred.float(), noise.float())
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

        if (step + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / 100
            print(f"  Step {step+1}: loss = {avg_loss:.6f}")

    # Save ControlNet LoRA weights
    controlnet.save_pretrained(model_dir)
    print(f"Saved repair model to {model_dir}")

    # Save training loss curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Repair Model Training Loss")
    fig.savefig(model_dir / "training_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Cleanup
    del text_encoder, vae, unet, controlnet, optimizer
    torch.cuda.empty_cache()


def test_repair_model(cfg: RI3DConfig):
    """Test the trained repair model on a few corrupted images."""
    from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        LCMScheduler,
    )
    from peft import PeftModel

    out_dir = cfg.scene_output_dir()
    test_dir = out_dir / "repair_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device
    dtype = cfg.dtype

    # Load training pairs for test input
    pairs = torch.load(out_dir / "repair_training_data" / "training_pairs.pt", weights_only=False)

    # Load base ControlNet + LoRA
    print("Loading repair pipeline...")
    controlnet = ControlNetModel.from_pretrained(cfg.controlnet_model, torch_dtype=dtype)
    controlnet = PeftModel.from_pretrained(controlnet, out_dir / "repair_model")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        cfg.sd_model, controlnet=controlnet, torch_dtype=dtype,
    ).to(device)

    # Use LCM scheduler for fast inference
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(cfg.lcm_lora)
    pipe.fuse_lora()

    # Disable safety checker for speed
    pipe.safety_checker = None

    # Test on a few pairs
    n_test = min(4, len(pairs))
    fig, axes = plt.subplots(3, n_test, figsize=(5 * n_test, 15))

    for j in range(n_test):
        corrupted, clean = pairs[j * (len(pairs) // n_test)]

        # Corrupted as PIL
        corrupted_pil = Image.fromarray(
            (corrupted.numpy() * 255).astype(np.uint8)
        ).resize((512, 512), Image.LANCZOS)

        # Run repair
        with torch.no_grad():
            result = pipe(
                prompt="",
                image=corrupted_pil,
                num_inference_steps=cfg.lcm_inference_steps,
                guidance_scale=cfg.lcm_guidance_scale,
            ).images[0]

        # Visualize
        axes[0, j].imshow(corrupted.numpy())
        axes[0, j].set_title("Corrupted")
        axes[0, j].axis("off")

        axes[1, j].imshow(result)
        axes[1, j].set_title("Repaired")
        axes[1, j].axis("off")

        axes[2, j].imshow(clean.numpy())
        axes[2, j].set_title("Ground Truth")
        axes[2, j].axis("off")

    fig.suptitle("Repair Model Test Results")
    fig.savefig(test_dir / "repair_test_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    del pipe
    torch.cuda.empty_cache()

    print(f"Saved repair test results to {test_dir}")


def run_step5(cfg: RI3DConfig):
    generate_leave_one_out_data(cfg)
    train_repair_model(cfg)
    test_repair_model(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: Train repair model")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene image directory")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    cfg = RI3DConfig(scene_dir=Path(args.scene), output_dir=Path(args.output))
    run_step5(cfg)
