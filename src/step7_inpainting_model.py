"""Step 7: Train the Inpainting Model (SD Inpainting + LoRA, RealFill-style).

Trains a SINGLE inpainting model on images from ALL scenes.

Per the paper (Sec 4.2, 8.2):
  - Fine-tune SD inpainting model on input images via random masking (RealFill approach)
  - This personalizes the model so hallucinated content matches the visual style
  - Images are resized so smallest dim = 512, then random 512x512 crop
  - Fine-tune for ~2000 iterations (scaled by number of scenes)

Outputs:
  - outputs/_shared_models/inpainting_model/  shared LoRA weights
  - outputs/<scene>/inpainting_test/          test results per scene
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


def generate_random_mask(H: int, W: int, min_ratio: float = 0.1,
                          max_ratio: float = 0.6) -> np.ndarray:
    """Generate a random rectangular mask for training.
    Returns binary mask where 1 = masked (to inpaint), 0 = visible.
    """
    mask = np.zeros((H, W), dtype=np.float32)
    n_rects = random.randint(1, 4)
    for _ in range(n_rects):
        h = int(H * random.uniform(0.1, 0.8))
        w = int(W * random.uniform(0.1, 0.8))
        y = random.randint(0, H - h)
        x = random.randint(0, W - w)
        mask[y:y+h, x:x+w] = 1.0
    return mask


def collect_all_scene_images(cfg: RI3DConfig) -> list[Image.Image]:
    """Collect input images from all scenes, resized for diffusion training.

    Per paper: resize so smallest dim = 512, then random 512x512 crop during training.
    """
    scenes = cfg.list_scenes()
    all_images = []

    for scene_dir in scenes:
        scene_cfg = RI3DConfig(scene_dir=scene_dir, output_dir=cfg.output_dir)
        paths_file = scene_cfg.scene_output_dir() / "image_paths.pt"
        if not paths_file.exists():
            print(f"  Skipping {scene_dir.name}: no image_paths.pt (run steps 1-4 first)")
            continue
        image_paths = torch.load(paths_file, weights_only=False)

        for ip in image_paths:
            img = Image.open(ip).convert("RGB")
            w, h = img.size
            if h < w:
                new_h, new_w = 512, int(w * 512 / h)
            else:
                new_w, new_h = 512, int(h * 512 / w)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            all_images.append(img)

        print(f"  {scene_dir.name}: {len(image_paths)} images")

    print(f"Total images for inpainting training: {len(all_images)}")
    return all_images


def train_inpainting_model(cfg: RI3DConfig):
    """Train a SINGLE inpainting model on images from ALL scenes."""
    from diffusers import (
        AutoencoderKL,
        UNet2DConditionModel,
        DDPMScheduler,
    )
    from transformers import CLIPTextModel, CLIPTokenizer
    from peft import LoraConfig, get_peft_model

    model_dir = cfg.shared_model_dir() / "inpainting_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device
    dtype = cfg.dtype

    # Collect images from all scenes
    all_images = collect_all_scene_images(cfg)
    if len(all_images) == 0:
        raise RuntimeError("No images found. Run steps 1-4 on all scenes first.")

    # Load SD inpainting (9-channel UNet)
    print("Loading SD 1.5 inpainting model...")
    inpaint_model_id = "runwayml/stable-diffusion-inpainting"

    tokenizer = CLIPTokenizer.from_pretrained(inpaint_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(inpaint_model_id, subfolder="text_encoder",
                                                  torch_dtype=dtype).to(device)
    vae = AutoencoderKL.from_pretrained(inpaint_model_id, subfolder="vae",
                                         torch_dtype=dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(inpaint_model_id, subfolder="unet",
                                                 torch_dtype=dtype).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(inpaint_model_id, subfolder="scheduler")

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    lora_config = LoraConfig(
        r=cfg.inpainting_lora_rank,
        lora_alpha=cfg.inpainting_lora_rank,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=cfg.inpainting_lr, weight_decay=1e-2)

    with torch.no_grad():
        tokens = tokenizer("", padding="max_length", max_length=77,
                           return_tensors="pt").input_ids.to(device)
        text_embeds = text_encoder(tokens)[0]

    # Scale iterations by number of scenes
    n_scenes = len(cfg.list_scenes())
    n_iters = max(cfg.inpainting_train_iters, cfg.inpainting_train_iters * n_scenes // 2)
    print(f"Training inpainting LoRA for {n_iters} iterations on {len(all_images)} images...")
    losses = []

    for step in tqdm(range(n_iters), desc="Inpainting training"):
        img_pil = random.choice(all_images)
        w, h = img_pil.size

        # Random 512x512 crop
        x0 = random.randint(0, max(0, w - 512))
        y0 = random.randint(0, max(0, h - 512))
        crop = img_pil.crop((x0, y0, x0 + 512, y0 + 512))
        img_np = np.array(crop).astype(np.float32) / 255.0

        mask_np = generate_random_mask(512, 512)

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device, dtype)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device, dtype)
        masked_img = img_tensor * (1 - mask_tensor)

        with torch.no_grad():
            clean_latent = vae.encode(img_tensor * 2 - 1).latent_dist.sample() * vae.config.scaling_factor
            masked_latent = vae.encode(masked_img * 2 - 1).latent_dist.sample() * vae.config.scaling_factor

        mask_latent = F.interpolate(mask_tensor, size=clean_latent.shape[-2:], mode="nearest")

        noise = torch.randn_like(clean_latent)
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,),
                          device=device).long()
        noisy_latent = noise_scheduler.add_noise(clean_latent, noise, t)

        unet_input = torch.cat([noisy_latent, mask_latent, masked_latent], dim=1)
        noise_pred = unet(unet_input, t, encoder_hidden_states=text_embeds).sample

        loss = F.mse_loss(noise_pred.float(), noise.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        if (step + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / 100
            print(f"  Step {step+1}: loss = {avg_loss:.6f}")

    unet.save_pretrained(model_dir)
    print(f"Saved shared inpainting model to {model_dir}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Inpainting Model Training Loss (all scenes)")
    fig.savefig(model_dir / "training_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    del text_encoder, vae, unet, optimizer
    torch.cuda.empty_cache()


def test_inpainting_model(cfg: RI3DConfig):
    """Test inpainting model on this scene's images."""
    from diffusers import StableDiffusionInpaintPipeline, LCMScheduler
    from peft import PeftModel

    # Find an actual scene to test on (when --train_models is used, cfg.scene_dir
    # may point to the dataset root rather than a specific scene)
    out_dir = cfg.scene_output_dir()
    image_paths_file = out_dir / "image_paths.pt"
    if not image_paths_file.exists():
        # Try to find any prepared scene in the dataset
        scenes = cfg.list_scenes()
        found = False
        for scene_dir in scenes:
            scene_out = cfg.output_dir / scene_dir.name
            if (scene_out / "image_paths.pt").exists():
                out_dir = scene_out
                image_paths_file = out_dir / "image_paths.pt"
                found = True
                break
        if not found:
            print("No prepared scenes found, skipping inpainting test.")
            return

    test_dir = out_dir / "inpainting_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device
    dtype = cfg.dtype

    model_dir = cfg.shared_model_dir() / "inpainting_model"
    print("Loading inpainting pipeline for test...")
    inpaint_model_id = "runwayml/stable-diffusion-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        inpaint_model_id, torch_dtype=dtype
    ).to(device)

    pipe.unet = PeftModel.from_pretrained(pipe.unet, model_dir)
    pipe.unet = pipe.unet.merge_and_unload()  # merge scene LoRA before adding LCM LoRA
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(cfg.lcm_lora, adapter_name="lcm")
    pipe.safety_checker = None

    image_paths = torch.load(image_paths_file, weights_only=False)
    n_test = min(3, len(image_paths))

    fig, axes = plt.subplots(3, n_test, figsize=(6 * n_test, 18))
    if n_test == 1:
        axes = axes.reshape(-1, 1)

    for j in range(n_test):
        img = Image.open(image_paths[j]).convert("RGB").resize((512, 512), Image.LANCZOS)
        mask_np = generate_random_mask(512, 512, min_ratio=0.15, max_ratio=0.4)
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")

        with torch.no_grad():
            result = pipe(
                prompt="",
                image=img,
                mask_image=mask_pil,
                num_inference_steps=cfg.lcm_inference_steps,
                guidance_scale=cfg.lcm_guidance_scale,
            ).images[0]

        img_np = np.array(img)
        masked_np = img_np.copy()
        masked_np[mask_np > 0.5] = [128, 128, 128]

        axes[0, j].imshow(img_np)
        axes[0, j].set_title("Original")
        axes[0, j].axis("off")
        axes[1, j].imshow(masked_np)
        axes[1, j].set_title("Masked")
        axes[1, j].axis("off")
        axes[2, j].imshow(result)
        axes[2, j].set_title("Inpainted")
        axes[2, j].axis("off")

    fig.suptitle(f"Inpainting Test — {cfg.scene_name}")
    fig.savefig(test_dir / "inpainting_test_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    del pipe
    torch.cuda.empty_cache()
    print(f"Saved inpainting test results to {test_dir}")


def run_step7(cfg: RI3DConfig):
    train_inpainting_model(cfg)
    test_inpainting_model(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 7: Train inpainting model")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene dir")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset root")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--n_views", type=int, default=3, help="Number of input views")
    args = parser.parse_args()

    cfg = RI3DConfig(scene_dir=Path(args.scene), dataset_dir=Path(args.dataset),
                     output_dir=Path(args.output), n_views=args.n_views)
    run_step7(cfg)
