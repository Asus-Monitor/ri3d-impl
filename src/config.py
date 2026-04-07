"""Global configuration for RI3D pipeline."""
from dataclasses import dataclass, field
from pathlib import Path
import torch

# Project root: parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class RI3DConfig:
    # Paths (defaults are relative to project root, not CWD)
    scene_dir: Path = None
    dataset_dir: Path = None
    output_dir: Path = None

    # View selection (int for count, or comma-separated filenames)
    n_views: int | str = 3

    # Device
    device: str = "cuda"
    dtype: torch.dtype = torch.float16

    # Image size (diffusion models operate at 512x512)
    diffusion_size: int = 512
    render_size: tuple = (512, 512)  # H, W for 3DGS rendering

    # DUSt3R
    dust3r_model: str = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    dust3r_confidence_threshold: float = 3.0  # log-confidence threshold for high-conf mask

    # Depth Anything V2
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"

    # Depth fusion (Poisson blending)
    poisson_lambda: float = 10.0  # gradient term weight (Eq. 2)
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    # Gaussian initialization
    gaussian_scale_factor: float = 1.4  # scale relative to pixel size
    gaussian_init_opacity: float = 0.1

    # Repair model (ControlNet + UNet LoRA, DPM++ scheduler)
    sd_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    controlnet_model: str = "lllyasviel/control_v11f1e_sd15_tile"
    repair_train_iters: int = 2000 # paper Sec 8.1: "fine-tune for 1800 iterations"
    repair_lr: float = 1e-4
    repair_lora_rank: int = 128
    repair_inference_steps: int = 20   # DPM++ 2M Karras
    repair_guidance_scale: float = 4.0  # moderate CFG — needed to make negative prompt effective
    repair_controlnet_scale: float = 0.75  # <1.0 lets LoRA override artifacts ControlNet would preserve
    repair_strength: float = 0.95         # img2img: 80% noise drowns artifacts, 20% signal preserves colors/layout
    repair_positive_prompt: str = "best quality, sharp detail"
    repair_negative_prompt: str = "blur, lowres, artifacts, distortion, worst quality"

    # Inpainting inference (LCM for speed)
    lcm_lora: str = "latent-consistency/lcm-lora-sdv1-5"
    inpainting_inference_steps: int = 8   # LCM steps
    inpainting_guidance_scale: float = 1.0   # no CFG — empty prompt means no text direction
    inpainting_strength: float = 1.0  # full noise in masked regions for maximum generation freedom

    # Leave-one-out training for repair
    loo_initial_iters: int = 6000  # iters before re-adding left-out view
    loo_total_iters: int = 10000   # total iters for leave-one-out 3DGS
    loo_snapshot_interval: int = 500   # snapshot corrupted renders every N iters after re-add
    loo_render_scale: float = 0.5     # train LOO 3DGS at lower res for speed (snapshots use full res)

    # Inpainting model (RealFill-style, per paper Sec 8.2)
    # Paper uses full UNet fine-tune; we use LoRA (attention + conv) for limited VRAM.
    inpainting_train_iters: int = 500 # paper Sec 8.2: "fine-tune the model for 2000 iterations"
    inpainting_lr: float = 2e-4        # RealFill standard UNet LoRA lr
    inpainting_lora_rank: int = 48     # attention + conv LoRA to approximate full fine-tune

    # Stage 1 optimization
    stage1_max_iters: int = 4000
    stage1_num_novel_views: int = 8
    stage1_refresh_interval: int = 400  # refresh repaired pseudo GT every N iters
    stage1_lr_position: float = 1.6e-4
    stage1_lr_opacity: float = 0.05
    stage1_lr_scaling: float = 5e-3
    stage1_lr_rotation: float = 1e-3
    stage1_lr_sh: float = 2.5e-3

    # Stage 2 optimization
    stage2_max_iters: int = 4000
    stage2_num_novel_views: int = 10
    stage2_num_inpaint_views: int = 5  # K views to inpaint per cycle
    stage2_inpaint_interval: int = 200
    stage2_inpaint_cutoff: int = 2800  # stop inpainting after this iter, only repair

    # Loss weights
    loss_l1_weight: float = 0.8
    loss_ssim_weight: float = 0.2
    loss_lpips_weight: float = 0.1
    loss_depth_corr_weight: float = 0.1
    loss_opacity_reg_weight: float = 0.01

    # Plateau detection (replaces fixed iteration counts)
    plateau_window: int = 400      # match refresh interval so window spans a full cycle
    plateau_threshold: float = 5e-4  # min relative improvement to continue
    plateau_min_iters: int = 2000   # run at least half the max iters before allowing early stop

    # 3DGS densification (scaled for 4000-iter optimization)
    densify_start: int = 500
    densify_stop: int = 2000
    densify_interval: int = 100
    densify_grad_threshold: float = 0.0002
    densify_reset_every: int = 1000  # reset opacities to allow pruning

    # Checkpointing
    checkpoint_dir: Path = None

    # Background mask clustering
    bg_mask_n_clusters: int = 2  # agglomerative clustering on depth

    def __post_init__(self):
        # Apply defaults relative to project root
        if self.dataset_dir is None:
            self.dataset_dir = PROJECT_ROOT / "dataset"
        if self.scene_dir is None:
            self.scene_dir = self.dataset_dir / "garden"
        if self.output_dir is None:
            self.output_dir = PROJECT_ROOT / "output"
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.output_dir / "checkpoints"

        # Resolve all paths to absolute (so CWD doesn't matter)
        self.scene_dir = Path(self.scene_dir).resolve()
        self.dataset_dir = Path(self.dataset_dir).resolve()
        self.output_dir = Path(self.output_dir).resolve()
        self.checkpoint_dir = Path(self.checkpoint_dir).resolve()

    @property
    def scene_name(self) -> str:
        return self.scene_dir.name

    def scene_output_dir(self) -> Path:
        d = self.output_dir / self.scene_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def shared_model_dir(self) -> Path:
        """Directory for models trained across all scenes."""
        d = self.output_dir / "_shared_models"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def load_image_paths(self) -> list[str]:
        """Load image_paths.pt for this scene."""
        import torch
        return torch.load(self.scene_output_dir() / "image_paths.pt", weights_only=False)

    def list_scenes(self) -> list[Path]:
        """List all scene subdirectories in dataset_dir."""
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        scenes = []
        for d in sorted(self.dataset_dir.iterdir()):
            if d.is_dir():
                has_images = any(p.suffix.lower() in exts for p in d.iterdir())
                if has_images:
                    scenes.append(d)
        return scenes
