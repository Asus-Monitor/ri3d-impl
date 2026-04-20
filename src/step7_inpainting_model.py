"""Step 7: Train the Inpainting Model (full SD Inpainting UNet fine-tuning, RealFill-style).

Per the paper (Sec 4.2, 8.2):
  - Fine-tune SD inpainting model on the scene's input images via random masking
  - This personalizes the model so hallucinated content matches the scene's visual style
  - Images are resized so smallest dim = 512, then random 512x512 crop
  - Fine-tune for 2000 iterations per scene
  - Full UNet fine-tuning following RealFill (Tang et al. / Nguyen implementation)

Outputs:
  - outputs/<scene>/inpainting_model/   per-scene fine-tuned UNet weights
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

random.seed(42)

def generate_random_mask(H: int, W: int, min_ratio: float = 0.15,
                          max_ratio: float = 0.5) -> np.ndarray:
    """Generate a random rectangular mask for testing.
    Returns binary mask where 1 = masked (to inpaint), 0 = visible.
    """
    mask = np.zeros((H, W), dtype=np.float32)
    n_rects = random.randint(1, 4)
    for _ in range(n_rects):
        h = int(H * random.uniform(min_ratio, max_ratio))
        w = int(W * random.uniform(min_ratio, max_ratio))
        y = random.randint(0, H - h)
        x = random.randint(0, W - w)
        mask[y:y+h, x:x+w] = 1.0
    return mask


def collect_scene_images(cfg: RI3DConfig) -> list[Image.Image]:
    """Collect this scene's input images, resized for diffusion training.

    Per paper: resize so smallest dim = 512, then random 512x512 crop during training.
    """
    image_paths = cfg.load_image_paths()
    images = []

    for ip in image_paths:
        img = Image.open(ip).convert("RGB")
        w, h = img.size
        if h < w:
            new_h, new_w = 512, int(w * 512 / h)
        else:
            new_w, new_h = 512, int(h * 512 / w)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        images.append(img)

    print(f"Scene {cfg.scene_name}: {len(images)} images for inpainting training")
    return images


def train_inpainting_model(cfg: RI3DConfig, shared_components=None):
    """Train a per-scene inpainting model (full UNet fine-tuning) on this scene's images.

    Per the paper (Sec 4.2, 8.2): fine-tune the Stable Diffusion inpainting model
    following RealFill methodology (Tang et al.). All UNet parameters are trained;
    VAE and text encoder stay frozen. Gradient checkpointing keeps memory tractable.

    Args:
        cfg: scene config
        shared_components: optional dict with pre-loaded frozen SD inpainting components
            (vae, text_embeds, noise_scheduler) to avoid reloading per scene
    """
    from diffusers import (
        AutoencoderKL,
        UNet2DConditionModel,
        DDPMScheduler,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    model_dir = cfg.scene_output_dir() / "inpainting_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device
    dtype = cfg.dtype
    inpaint_model_id = "runwayml/stable-diffusion-inpainting"

    
    scene_images = collect_scene_images(cfg)
    if len(scene_images) == 0:
        raise RuntimeError(f"No images for {cfg.scene_name}. Run steps 1-4 first.")

    
    _owns_components = shared_components is None
    if _owns_components:
        print("Loading SD 1.5 inpainting model...")
        tokenizer = CLIPTokenizer.from_pretrained(inpaint_model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(inpaint_model_id, subfolder="text_encoder",
                                                      torch_dtype=dtype).to(device)
        vae = AutoencoderKL.from_pretrained(inpaint_model_id, subfolder="vae",
                                             torch_dtype=dtype).to(device)
        noise_scheduler = DDPMScheduler.from_pretrained(inpaint_model_id, subfolder="scheduler")
        text_encoder.requires_grad_(False)
        vae.requires_grad_(False)
        with torch.no_grad():
            tokens = tokenizer("", padding="max_length", max_length=77,
                               return_tensors="pt").input_ids.to(device)
            text_embeds = text_encoder(tokens)[0]
        
        del text_encoder, tokenizer
        torch.cuda.empty_cache()
    else:
        vae = shared_components["vae"]
        text_embeds = shared_components["text_embeds"]
        noise_scheduler = shared_components["noise_scheduler"]

    
    
    
    
    
    
    
    torch.backends.cudnn.benchmark = True
    vae.eval()
    latent_crop = 64  
    scale = vae.config.scaling_factor

    img_tensors_gpu = []    
    clean_latents_gpu = []  

    with torch.no_grad():
        for img_pil in scene_images:
            w, h = img_pil.size
            h8, w8 = (h // 8) * 8, (w // 8) * 8
            if (h8, w8) != (h, w):
                img_pil = img_pil.crop((0, 0, w8, h8))
            img_np = np.array(img_pil).astype(np.float32) / 255.0
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).to(device, dtype)
            img_tensors_gpu.append(img_t)
            lat = vae.encode(img_t.unsqueeze(0) * 2 - 1).latent_dist.sample() * scale
            clean_latents_gpu.append(lat.squeeze(0))

        
        
        n_orig = len(img_tensors_gpu)
        for i in range(n_orig):
            img_flip = img_tensors_gpu[i].flip(-1)
            img_tensors_gpu.append(img_flip)
            lat_flip = vae.encode(img_flip.unsqueeze(0) * 2 - 1).latent_dist.sample() * scale
            clean_latents_gpu.append(lat_flip.squeeze(0))

    n_images = len(img_tensors_gpu)
    print(f"  {n_orig} originals + {n_orig} flips = {n_images} effective training images")

    
    
    
    
    # FP16 model + 8-bit Adam: keeps everything on GPU with ~3x less
    # optimizer memory than FP32 Adam.
    unet = UNet2DConditionModel.from_pretrained(inpaint_model_id, subfolder="unet",
                                                 torch_dtype=dtype).to(device)
    unet.requires_grad_(True)
    unet.enable_gradient_checkpointing()
    # Channels-last NHWC: ~5-10% faster on Turing+ conv kernels, no math change.
    unet.to(memory_format=torch.channels_last)
    unet.train()
    n_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"Full UNet fine-tuning: {n_params:,} trainable parameters")

    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=cfg.inpainting_lr, weight_decay=1e-2)

    n_iters = cfg.inpainting_train_iters
    max_t = noise_scheduler.config.num_train_timesteps

    
    gpu_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    if gpu_mem_gb >= 24:
        batch_size = 2
    else:
        batch_size = 1
    n_steps = (n_iters + batch_size - 1) // batch_size
    text_embeds_b = text_embeds.expand(batch_size, -1, -1)

    print(f"Training inpainting UNet (full fine-tune) for {n_iters} iterations "
          f"(batch={batch_size}) on {n_images} images...")
    losses = []
    iter_count = 0

    for step in tqdm(range(n_steps), desc=f"Inpainting [{cfg.scene_name}]"):
        actual_bs = min(batch_size, n_iters - iter_count)
        if actual_bs <= 0:
            break

        # Inlined bs=1 fast path (the 8 GB regime).
        if actual_bs == 1:
            idx = random.randint(0, n_images - 1)
            img_t = img_tensors_gpu[idx]
            full_lat = clean_latents_gpu[idx]
            _, h, w = img_t.shape
            x0 = random.randint(0, max(0, (w - 512) // 8)) * 8
            y0 = random.randint(0, max(0, (h - 512) // 8)) * 8
            lx, ly = x0 // 8, y0 // 8
            crop = img_t[:, y0:y0+512, x0:x0+512].unsqueeze(0)
            clean_lat_b = full_lat[:, ly:ly+latent_crop, lx:lx+latent_crop].unsqueeze(0)

            mask = torch.zeros(1, 1, 512, 512, device=device, dtype=dtype)
            for _r in range(random.randint(1, 4)):
                rh = int(512 * random.uniform(0.1, 0.8))
                rw = int(512 * random.uniform(0.1, 0.8))
                ry = random.randint(0, 512 - rh)
                rx = random.randint(0, 512 - rw)
                mask[:, :, ry:ry+rh, rx:rx+rw] = 1.0

            masked_img_b = crop * (1 - mask)
            mask_lat_b = F.interpolate(mask, size=(latent_crop, latent_crop), mode="nearest")
            noise_b = torch.randn_like(clean_lat_b)
            t_b = torch.randint(0, max_t, (1,), device=device).long()
        else:
            clean_lats, masked_imgs, mask_lats, noises, ts = [], [], [], [], []
            for _ in range(actual_bs):
                idx = random.randint(0, n_images - 1)
                img_t = img_tensors_gpu[idx]
                full_lat = clean_latents_gpu[idx]
                _, h, w = img_t.shape
                x0 = random.randint(0, max(0, (w - 512) // 8)) * 8
                y0 = random.randint(0, max(0, (h - 512) // 8)) * 8
                lx, ly = x0 // 8, y0 // 8
                crop = img_t[:, y0:y0+512, x0:x0+512].unsqueeze(0)
                cl = full_lat[:, ly:ly+latent_crop, lx:lx+latent_crop].unsqueeze(0)
                mask = torch.zeros(1, 1, 512, 512, device=device, dtype=dtype)
                for _r in range(random.randint(1, 4)):
                    rh = int(512 * random.uniform(0.1, 0.8))
                    rw = int(512 * random.uniform(0.1, 0.8))
                    ry = random.randint(0, 512 - rh)
                    rx = random.randint(0, 512 - rw)
                    mask[:, :, ry:ry+rh, rx:rx+rw] = 1.0
                masked_imgs.append(crop * (1 - mask))
                mask_lats.append(F.interpolate(mask, size=(latent_crop, latent_crop), mode="nearest"))
                clean_lats.append(cl)
                noises.append(torch.randn_like(cl))
                ts.append(torch.randint(0, max_t, (1,), device=device).long())
            clean_lat_b = torch.cat(clean_lats)
            masked_img_b = torch.cat(masked_imgs)
            mask_lat_b = torch.cat(mask_lats)
            noise_b = torch.cat(noises)
            t_b = torch.cat(ts)

        with torch.no_grad():
            masked_lat_b = vae.encode(masked_img_b * 2 - 1).latent_dist.sample() * scale

        noisy_lat_b = noise_scheduler.add_noise(clean_lat_b, noise_b, t_b)
        unet_input = torch.cat([noisy_lat_b, mask_lat_b, masked_lat_b], dim=1).contiguous(
            memory_format=torch.channels_last)

        # No autocast: model + tensors already fp16.
        noise_pred = unet(unet_input, t_b, encoder_hidden_states=text_embeds_b[:actual_bs]).sample

        loss = F.mse_loss(noise_pred.float(), noise_b.float())
        loss.backward()

        # Speed: skip .isfinite() guard (forces sync); just clip + step.
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        iter_count += actual_bs
        losses.append(loss.detach())  # GPU tensor; sync deferred to print
        if iter_count % 100 < batch_size or step == n_steps - 1:
            recent_t = torch.stack(losses[-max(1, 100 // batch_size):]).float()
            avg_loss = recent_t.mean().item()
            print(f"  Step ~{min(iter_count, n_iters)}: loss = {avg_loss:.6f}")

    unet.save_pretrained(model_dir)
    print(f"Saved inpainting model to {model_dir}")

    losses_cpu = torch.stack(losses).float().cpu().numpy() if losses else []
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses_cpu)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Inpainting Model Training Loss — {cfg.scene_name}")
    fig.savefig(model_dir / "training_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    del unet, optimizer, img_tensors_gpu, clean_latents_gpu, losses
    if _owns_components:
        del vae
    torch.cuda.empty_cache()


def test_inpainting_model(cfg: RI3DConfig):
    """Test inpainting model with multiple mask positions per image."""
    from diffusers import (
        StableDiffusionInpaintPipeline,
        UNet2DConditionModel,
        DPMSolverMultistepScheduler,
    )

    out_dir = cfg.scene_output_dir()
    model_dir = out_dir / "inpainting_model"
    if not model_dir.exists():
        print(f"No inpainting model for {cfg.scene_name}, skipping test.")
        return

    test_dir = out_dir / "inpainting_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device
    dtype = cfg.dtype

    print(f"Loading inpainting pipeline for test ({cfg.scene_name})...")
    inpaint_model_id = "runwayml/stable-diffusion-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        inpaint_model_id, torch_dtype=dtype
    ).to(device)

    pipe.unet = UNet2DConditionModel.from_pretrained(
        model_dir, torch_dtype=dtype).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True)
    pipe.safety_checker = None

    image_paths = cfg.load_image_paths()
    n_images = len(image_paths)

    
    masks_per_image = 3
    total_cols = n_images * masks_per_image
    fig, axes = plt.subplots(3, total_cols, figsize=(5 * total_cols, 15))
    if total_cols == 1:
        axes = axes.reshape(-1, 1)

    mask_configs = [
        (0.10, 0.25),  
        (0.20, 0.40),  
        (0.35, 0.60),  
    ]

    from utils import prepare_for_pipeline

    for i in range(n_images):
        img_orig = Image.open(image_paths[i]).convert("RGB")
        img_resized, pipe_h, pipe_w = prepare_for_pipeline(img_orig)
        img_np = np.array(img_resized)

        for j, (min_r, max_r) in enumerate(mask_configs):
            col = i * masks_per_image + j
            mask_np = generate_random_mask(pipe_h, pipe_w, min_ratio=min_r, max_ratio=max_r)
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")

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

            masked_np = img_np.copy()
            masked_np[mask_np > 0.5] = [128, 128, 128]
            size_label = ["small", "medium", "large"][j]

            axes[0, col].imshow(img_np)
            axes[0, col].set_title(f"Original (img {i})")
            axes[0, col].axis("off")
            axes[1, col].imshow(masked_np)
            axes[1, col].set_title(f"Masked ({size_label})")
            axes[1, col].axis("off")
            axes[2, col].imshow(result)
            axes[2, col].set_title(f"Inpainted ({size_label})")
            axes[2, col].axis("off")

    fig.suptitle(f"Inpainting Test — {cfg.scene_name}", fontsize=16)
    fig.savefig(test_dir / "inpainting_test_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    del pipe
    torch.cuda.empty_cache()
    print(f"Saved inpainting test results to {test_dir}")


def run_step7(cfg: RI3DConfig, shared_components=None):
    """Full step 7 for a single scene: train and test."""
    train_inpainting_model(cfg, shared_components=shared_components)
    test_inpainting_model(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 7: Train inpainting model")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene dir")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset root")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--n_views", type=str, default="3", help="Number of input views per scene")
    args = parser.parse_args()

    cfg = RI3DConfig(scene_dir=Path(args.scene),
                     dataset_dir=Path(args.dataset) if args.dataset else None,
                     output_dir=Path(args.output) if args.output else None,
                     n_views=args.n_views)
    run_step7(cfg)
