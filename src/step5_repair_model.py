"""Step 5: Train the Repair Model (ControlNet + LoRA + LCM).

This step has two phases:
  A) Generate leave-one-out training data for each scene (per-scene)
  B) Train a SINGLE repair model on ALL scenes' data combined (shared)

Per the paper (Sec 4.2, 8.1):
  1. Leave-one-out: create N subsets of N-1 images per scene
  2. Train N small 3DGS representations (6000 iters without, then to 10000 with left-out)
  3. Snapshot intermediate renders -> training pairs (corrupted, clean)
  4. Fine-tune ControlNet (tile) with LoRA on ALL pairs from ALL scenes for 1800 iters

Outputs:
  - outputs/<scene>/repair_training_data/  per-scene corrupted/clean pairs
  - outputs/_shared_models/repair_model/   shared fine-tuned ControlNet LoRA weights
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
    """Generate corrupted/clean image pairs for a single scene using leave-one-out."""
    from gsplat import rasterization, DefaultStrategy

    out_dir = cfg.scene_output_dir()
    data_dir = out_dir / "repair_training_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    image_paths = torch.load(out_dir / "image_paths.pt", weights_only=False)
    poses = torch.load(out_dir / "dust3r_poses.pt", weights_only=True).float()
    intrinsics = torch.load(out_dir / "dust3r_intrinsics.pt", weights_only=True).float()
    gaussians_init = torch.load(out_dir / "init_gaussians.pt", weights_only=True)
    n_images = len(image_paths)

    device = cfg.device
    fused_depth = torch.load(out_dir / "fused_depths" / "fused_depth_000.pt", weights_only=True)
    H, W = fused_depth.shape

    gt_images = []
    for ip in image_paths:
        img = Image.open(ip).convert("RGB").resize((W, H), Image.LANCZOS)
        gt_images.append(torch.from_numpy(np.array(img)).float() / 255.0)

    ssim_fn = SSIMLoss().to(device)
    all_pairs = []

    for left_out_idx in range(n_images):
        name = Path(image_paths[left_out_idx]).stem
        print(f"\nLeave-one-out: excluding view {left_out_idx} ({name})")

        train_indices = [j for j in range(n_images) if j != left_out_idx]
        clean_image = gt_images[left_out_idx].to(device)

        model = GaussianModel(gaussians_init, device)
        optimizers = model.setup_optimizers(cfg)
        strategy, strategy_state = model.setup_strategy(cfg, scene_scale=1.0)

        # Phase 1: Train without left-out view
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

        # Snapshot: render left-out view as early corrupted
        with torch.no_grad():
            w2c_lo = torch.linalg.inv(poses[left_out_idx]).to(device)
            K_lo = intrinsics[left_out_idx].to(device)
            r = model.render(w2c_lo, K_lo, H, W)
            corrupted_early = r["image"].clamp(0, 1).cpu()
            all_pairs.append((corrupted_early, gt_images[left_out_idx]))

        # Phase 2: Re-add left-out view, continue training
        # Paper (Sec 4.2): "by initially excluding an image and later reintroducing it,
        # we generate progressively refined corrupted images, improving the repair model's
        # ability to handle the final 3DGS optimization." The full spectrum from heavily
        # corrupted to nearly clean is intentional — teaches the model to handle all
        # corruption levels encountered during Stage 1 optimization.
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


def train_repair_model(cfg: RI3DConfig):
    """Train a SINGLE repair model on data from ALL scenes."""
    from diffusers import (
        ControlNetModel,
        DDPMScheduler,
        AutoencoderKL,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer
    from peft import LoraConfig, get_peft_model

    model_dir = cfg.shared_model_dir() / "repair_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device
    dtype = cfg.dtype

    # Collect training pairs from ALL scenes
    all_pairs = []
    scenes = cfg.list_scenes()
    for scene_dir in scenes:
        scene_cfg = RI3DConfig(scene_dir=scene_dir, output_dir=cfg.output_dir)
        pairs_path = scene_cfg.scene_output_dir() / "repair_training_data" / "training_pairs.pt"
        if pairs_path.exists():
            pairs = torch.load(pairs_path, weights_only=False)
            all_pairs.extend(pairs)
            print(f"  Loaded {len(pairs)} pairs from {scene_dir.name}")

    if len(all_pairs) == 0:
        raise RuntimeError("No training pairs found. Run data generation first.")
    print(f"Total training pairs across all scenes: {len(all_pairs)}")

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

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=cfg.repair_lora_rank,
        lora_alpha=cfg.repair_lora_rank,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"],
        lora_dropout=0.0,
    )
    controlnet = get_peft_model(controlnet, lora_config)
    controlnet.print_trainable_parameters()

    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=cfg.repair_lr, weight_decay=1e-2)

    with torch.no_grad():
        tokens = tokenizer("", padding="max_length", max_length=77,
                           return_tensors="pt").input_ids.to(device)
        text_embeds = text_encoder(tokens)[0]

    # Scale iterations by number of scenes (more data = more iters)
    n_iters = max(cfg.repair_train_iters, cfg.repair_train_iters * len(scenes) // 2)
    print(f"Training ControlNet LoRA for {n_iters} iterations on {len(all_pairs)} pairs...")
    losses = []

    for step in tqdm(range(n_iters), desc="Repair training"):
        corrupted, clean = random.choice(all_pairs)

        corrupted_512 = F.interpolate(
            corrupted.permute(2, 0, 1).unsqueeze(0), size=(512, 512),
            mode="bilinear", align_corners=False
        ).to(device, dtype)

        clean_512 = F.interpolate(
            clean.permute(2, 0, 1).unsqueeze(0), size=(512, 512),
            mode="bilinear", align_corners=False
        ).to(device, dtype)

        with torch.no_grad():
            clean_latent = vae.encode(clean_512 * 2 - 1).latent_dist.sample() * vae.config.scaling_factor

        noise = torch.randn_like(clean_latent)
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,),
                          device=device).long()
        noisy_latent = noise_scheduler.add_noise(clean_latent, noise, t)

        controlnet_output = controlnet(
            noisy_latent, t, encoder_hidden_states=text_embeds,
            controlnet_cond=corrupted_512,
            return_dict=False,
        )
        down_block_res_samples = controlnet_output[0]
        mid_block_res_sample = controlnet_output[1]

        noise_pred = unet(
            noisy_latent, t, encoder_hidden_states=text_embeds,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample

        loss = F.mse_loss(noise_pred.float(), noise.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        if (step + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / 100
            print(f"  Step {step+1}: loss = {avg_loss:.6f}")

    controlnet.save_pretrained(model_dir)
    print(f"Saved shared repair model to {model_dir}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Repair Model Training Loss (all scenes)")
    fig.savefig(model_dir / "training_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    del text_encoder, vae, unet, controlnet, optimizer
    torch.cuda.empty_cache()


def test_repair_model(cfg: RI3DConfig):
    """Test the trained repair model on corrupted images from this scene."""
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

    pairs_path = out_dir / "repair_training_data" / "training_pairs.pt"
    if not pairs_path.exists():
        print("No training pairs for this scene, skipping test.")
        return
    pairs = torch.load(pairs_path, weights_only=False)

    model_dir = cfg.shared_model_dir() / "repair_model"
    print("Loading repair pipeline for test...")
    controlnet = ControlNetModel.from_pretrained(cfg.controlnet_model, torch_dtype=dtype)
    controlnet = PeftModel.from_pretrained(controlnet, model_dir)
    controlnet = controlnet.merge_and_unload()

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        cfg.sd_model, controlnet=controlnet, torch_dtype=dtype,
    ).to(device)

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(cfg.lcm_lora)
    pipe.fuse_lora()
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
        ).resize((512, 512), Image.LANCZOS)

        with torch.no_grad():
            result = pipe(
                prompt="",
                image=corrupted_pil,
                num_inference_steps=cfg.lcm_inference_steps,
                guidance_scale=cfg.lcm_guidance_scale,
            ).images[0]

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


def run_step5(cfg: RI3DConfig):
    """Full step 5: generate data for all scenes, train shared model, test on this scene."""
    generate_all_scenes_data(cfg)
    train_repair_model(cfg)
    test_repair_model(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: Train repair model")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene dir (or dataset root)")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset root with all scenes")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--n_views", type=str, default="3", help="Number of input views per scene")
    parser.add_argument("--data_only", action="store_true", help="Only generate data, don't train")
    parser.add_argument("--train_only", action="store_true", help="Only train, assume data exists")
    args = parser.parse_args()

    cfg = RI3DConfig(scene_dir=Path(args.scene), dataset_dir=Path(args.dataset),
                     output_dir=Path(args.output), n_views=args.n_views)

    if args.data_only:
        generate_all_scenes_data(cfg)
    elif args.train_only:
        train_repair_model(cfg)
    else:
        run_step5(cfg)
