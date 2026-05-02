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

    # Repair model (full ControlNet fine-tune, DPM++ scheduler, img2img pipeline)
    sd_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    controlnet_model: str = "lllyasviel/control_v11f1e_sd15_tile"
    repair_train_iters: int = 2500
    repair_lr: float = 1e-4  # full FT — between LoRA's 1e-3 and the over-conservative 1e-5
    repair_inference_steps: int = 50  # DDIM steps per GaussianObject
    repair_guidance_scale: float = 1.0  # CFG off — trust the ControlNet (GaussianObject convention)
    repair_controlnet_scale: float = 1.0  # GaussianObject default; >1.0 over-fits to corrupted render
    repair_hflip_augment: bool = True
    repair_snr_gamma: float = 5.0  # Min-SNR-γ weighting; ε-pred form: min(snr, γ)/snr
    repair_test_lpips_success: float = 0.10  # test PASS if LPIPS_repaired <= this
    repair_strength: float = 0.6  # default img2img denoise strength (used by test)
    repair_strength_max: float = 1.0  # test/eval: full denoise to showcase repair power
    repair_strength_min: float = 0.1  # test/eval: minimal denoise on clean inputs

    # Per-refresh strength schedule: starts high (big corrections early when the
    # render is rough), decays low (small corrections late when render is close
    # to truth). Adaptive schedule supports the paper's iterative refresh: the
    # anti-spiral mechanism is opacity reg + live M_α shrinking under it, not
    # avoiding repeated repair calls.
    stage1_repair_strength_max: float = 0.5
    stage1_repair_strength_min: float = 0.1
    stage2_repair_strength_max: float = 0.3
    stage2_repair_strength_min: float = 0.05
    # DDIM eta = 1.0 (stochastic) matches GaussianObject's recipe and the
    # tested-passing configuration of `repair_test`. Deterministic (eta=0)
    # was the hallucination-spiral root cause: with a fixed per-view seed,
    # every refresh lands on the same local mode of the SD conditional — if
    # that mode is slightly stylized, 3DGS absorbs it, next refresh repeats,
    # stylization compounds. eta=1.0 samples from a broader posterior; the
    # 3DGS projection averages across modes and lands closer to the natural-
    # image manifold. The earlier rationale ("stochastic DDIM compounds
    # drift") had the mechanism backwards.
    repair_eta: float = 1.0
    # DreamBooth-style rare token + scene-name. The rare token forces the model
    # to bind scene-specific repair behavior to this identifier, instead of
    # diluting the generic "best quality" subspace.
    repair_scene_token: str = "xxy5syt00"
    repair_positive_prompt: str = "a photo of a xxy5syt00, best quality"

    # Inpainting inference (standard DPM++ scheduler)
    inpainting_inference_steps: int = 30   # DPM++ steps
    inpainting_guidance_scale: float = 1.0   # no CFG — empty prompt means no text direction
    inpainting_strength: float = 1.0  # full noise in masked regions for maximum generation freedom

    # Leave-one-out training for repair
    loo_initial_iters: int = 6000  # iters before re-adding left-out view
    loo_total_iters: int = 10000   # total iters for leave-one-out 3DGS
    loo_render_scale: float = 1.0     # LOO 3DGS training resolution scale (1.0 = full res)

    # Inpainting model (RealFill-style full UNet fine-tune, per paper Sec 4.2, 8.2)
    inpainting_train_iters: int = 2000 # paper Sec 8.2: "fine-tune the model for 2000 iterations"
    inpainting_lr: float = 2e-5        # full UNet fine-tuning lr

    # Stage 1 optimization
    # Capped at 1300: empirically, repair quality plateaus around step ~1300
    # and deteriorates beyond — successive refreshes inject more drift than
    # they fix once the easy corrections are made. With warmup=100 and
    # refresh_interval=400, range(1300) covers refreshes at steps 100, 500,
    # 900 (step 1300 is past max_iters so the 4th refresh doesn't fire).
    stage1_max_iters: int = 901
    stage1_num_novel_views: int = 8
    stage1_refresh_interval: int = 400  # paper §8.3: repaired views refreshed every 400 iters
    # Warmup: train on inputs only first, so that when we render at novel views
    # for the first repair call, the render is "mid-training 3DGS" style — the
    # distribution the repair model was trained on (§4.2 trains on renders from
    # 6000-10000 iter leave-one-out 3DGS). Step-0 renders from per-pixel init
    # are OOD for the repair model; feeding them in produces hallucinated
    # pseudo-GTs that anchor all subsequent optimization. Paper doesn't specify
    # this because for them repair's training distribution and inference-step-0
    # render distribution may have been close enough; for us they aren't.
    stage1_warmup_iters: int = 100
    stage1_lr_position: float = 1.6e-4
    stage1_lr_opacity: float = 0.05
    stage1_lr_scaling: float = 5e-3
    stage1_lr_rotation: float = 1e-3
    stage1_lr_sh: float = 2.5e-3

    # Stage 2 optimization
    stage2_max_iters: int = 4000
    stage2_num_novel_views: int = 10
    stage2_num_inpaint_views: int = 5  # K views to inpaint per cycle
    stage2_inpaint_interval: int = 200  # paper §8.3: sample & inpaint every 200 iters
    stage2_inpaint_cutoff: int = 2800  # stop inpainting after this iter, only repair

    # Loss weights
    loss_l1_weight: float = 0.8
    loss_ssim_weight: float = 0.2
    loss_lpips_weight: float = 0.1
    loss_depth_corr_weight: float = 0.1
    loss_opacity_reg_weight: float = 0.3  # anti-ratchet: with fixed capacity (no densification), this can now actually shrink alpha in background holes faster than repair inflates it

    # Plateau detection (replaces fixed iteration counts)
    plateau_window: int = 400      # match refresh interval so window spans a full cycle
    plateau_threshold: float = 5e-4  # min relative improvement to continue
    plateau_min_iters: int = 2000   # run at least half the max iters before allowing early stop

    # 3DGS densification. Disabled by default: paper §4.1 initializes one
    # Gaussian per pixel per input image (already ~500k for 3 images @ ~400x
    # resolution). §8.3 specifies no densification schedule, and the whole
    # point of the paper's dense init is that standard 3DGS densification —
    # designed for sparse COLMAP points ~10k — becomes redundant capacity
    # bloat. With 500k Gaussians already, aggressive densification lets the
    # representation fit hallucinated pseudo-GTs perfectly, defeating the
    # opacity reg anti-ratchet.
    densify_start: int = 99999  # never (paper uses dense init as capacity budget)
    densify_stop: int = 99999
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
