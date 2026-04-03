"""Shared 3DGS training utilities used across steps.

Provides:
  - GaussianModel: parameter container + rendering
  - Loss functions (L1, SSIM, LPIPS, depth correlation)
  - Basic training loop for leave-one-out and main optimization
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization, DefaultStrategy
import lpips as lpips_module

from config import RI3DConfig


class GaussianModel:
    """Container for 3D Gaussian parameters with rendering support."""

    def __init__(self, gaussians: dict, device: str = "cuda"):
        self.device = device
        self.params = nn.ParameterDict({
            "means":     nn.Parameter(gaussians["means"].float().to(device)),
            "scales":    nn.Parameter(gaussians["scales"].float().to(device)),
            "quats":     nn.Parameter(gaussians["quats"].float().to(device)),
            "opacities": nn.Parameter(gaussians["opacities"].float().to(device)),
            "colors":    nn.Parameter(gaussians["colors"].float().to(device)),
        })

    @property
    def n_gaussians(self):
        return self.params["means"].shape[0]

    def state_dict(self):
        return {k: v.data.clone() for k, v in self.params.items()}

    def load_state_dict(self, d):
        for k, v in d.items():
            self.params[k].data.copy_(v)

    def render(self, w2c: torch.Tensor, K: torch.Tensor, H: int, W: int,
               bg_color: torch.Tensor | None = None,
               return_depth: bool = False) -> dict:
        """Render from a single camera.

        Args:
            w2c: (4, 4) world-to-camera
            K: (3, 3) intrinsics
            H, W: image dimensions
            bg_color: (3,) background color or None (black)
            return_depth: if True, use RGB+D render mode

        Returns dict with 'image' (H,W,3), 'alpha' (H,W,1), optionally 'depth' (H,W,1)
        """
        viewmats = w2c.unsqueeze(0).to(self.device)
        Ks = K.unsqueeze(0).to(self.device)
        backgrounds = bg_color.unsqueeze(0) if bg_color is not None else None
        render_mode = "RGB+D" if return_depth else "RGB"

        render_colors, render_alphas, meta = rasterization(
            means=self.params["means"],
            quats=self.params["quats"],
            scales=torch.exp(self.params["scales"]),
            opacities=torch.sigmoid(self.params["opacities"]),
            colors=self.params["colors"],
            viewmats=viewmats,
            Ks=Ks,
            width=W, height=H,
            sh_degree=None,
            packed=True,
            near_plane=0.01,
            far_plane=1000.0,
            render_mode=render_mode,
            backgrounds=backgrounds,
        )

        result = {
            "image": render_colors[0, :, :, :3],    # (H, W, 3)
            "alpha": render_alphas[0],                # (H, W, 1)
            "meta": meta,
        }
        if return_depth:
            result["depth"] = render_colors[0, :, :, 3:4]  # (H, W, 1)

        return result

    def render_for_optim(self, w2c: torch.Tensor, K: torch.Tensor, H: int, W: int,
                          strategy: DefaultStrategy, strategy_state: dict, step: int,
                          bg_color: torch.Tensor | None = None) -> dict:
        """Render with densification hooks for optimization."""
        viewmats = w2c.unsqueeze(0).to(self.device)
        Ks = K.unsqueeze(0).to(self.device)
        backgrounds = bg_color.unsqueeze(0) if bg_color is not None else None

        render_colors, render_alphas, meta = rasterization(
            means=self.params["means"],
            quats=self.params["quats"],
            scales=torch.exp(self.params["scales"]),
            opacities=torch.sigmoid(self.params["opacities"]),
            colors=self.params["colors"],
            viewmats=viewmats,
            Ks=Ks,
            width=W, height=H,
            sh_degree=None,
            packed=True,
            near_plane=0.01, far_plane=1000.0,
            render_mode="RGB+D",
            backgrounds=backgrounds,
        )

        # Pre-backward hook
        strategy.step_pre_backward(self.params, self._optimizers, strategy_state, step, meta)

        return {
            "image": render_colors[0, :, :, :3],
            "alpha": render_alphas[0],
            "depth": render_colors[0, :, :, 3:4],
            "meta": meta,
        }

    def setup_optimizers(self, cfg: RI3DConfig) -> dict:
        """Create per-parameter optimizers."""
        self._optimizers = {
            "means":     torch.optim.Adam([self.params["means"]], lr=cfg.stage1_lr_position),
            "scales":    torch.optim.Adam([self.params["scales"]], lr=cfg.stage1_lr_scaling),
            "quats":     torch.optim.Adam([self.params["quats"]], lr=cfg.stage1_lr_rotation),
            "opacities": torch.optim.Adam([self.params["opacities"]], lr=cfg.stage1_lr_opacity),
            "colors":    torch.optim.Adam([self.params["colors"]], lr=cfg.stage1_lr_sh),
        }
        return self._optimizers

    def setup_strategy(self, cfg: RI3DConfig, scene_scale: float = 1.0):
        """Setup densification strategy."""
        self._strategy = DefaultStrategy(
            refine_start_iter=cfg.densify_start,
            refine_stop_iter=cfg.densify_stop,
            refine_every=cfg.densify_interval,
            grow_grad2d=cfg.densify_grad_threshold,
            verbose=False,
        )
        self._strategy.check_sanity(self.params, self._optimizers)
        self._strategy_state = self._strategy.initialize_state(scene_scale=scene_scale)
        return self._strategy, self._strategy_state

    def step_post_backward(self, step: int, meta: dict):
        """Post-backward densification step."""
        self._strategy.step_post_backward(
            self.params, self._optimizers, self._strategy_state, step, meta, packed=True
        )

    def optimizer_step(self):
        for opt in self._optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)


# --- Loss functions ---

class SSIMLoss(nn.Module):
    """Structural Similarity Index loss."""

    def __init__(self, window_size: int = 11, channel: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-coords**2 / (2 * sigma**2))
        g = g / g.sum()
        window = g.unsqueeze(1) @ g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0).expand(channel, 1, -1, -1)
        self.register_buffer("window", window)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """img1, img2: (H, W, 3) in [0,1]."""
        # Rearrange to (1, C, H, W)
        x = img1.permute(2, 0, 1).unsqueeze(0)
        y = img2.permute(2, 0, 1).unsqueeze(0)

        C = self.channel
        w = self.window.to(x.device, x.dtype)

        mu1 = F.conv2d(x, w, padding=self.window_size // 2, groups=C)
        mu2 = F.conv2d(y, w, padding=self.window_size // 2, groups=C)

        mu1_sq, mu2_sq = mu1**2, mu2**2
        mu12 = mu1 * mu2

        sigma1_sq = F.conv2d(x * x, w, padding=self.window_size // 2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(y * y, w, padding=self.window_size // 2, groups=C) - mu2_sq
        sigma12 = F.conv2d(x * y, w, padding=self.window_size // 2, groups=C) - mu12

        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1.0 - ssim_map.mean()


class LPIPSLoss:
    """Perceptual loss using LPIPS (SqueezeNet for low memory)."""

    def __init__(self, device="cuda"):
        self.loss_fn = lpips_module.LPIPS(net="squeeze").to(device).eval()
        for p in self.loss_fn.parameters():
            p.requires_grad_(False)

    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """img1, img2: (H, W, 3) in [0,1]."""
        x = img1.permute(2, 0, 1).unsqueeze(0) * 2 - 1  # to [-1, 1]
        y = img2.permute(2, 0, 1).unsqueeze(0) * 2 - 1
        return self.loss_fn(x, y).mean()


def depth_correlation_loss(rendered_depth: torch.Tensor, mono_depth: torch.Tensor,
                            mask: torch.Tensor | None = None) -> torch.Tensor:
    """Pearson correlation loss between rendered and monocular depth.

    Args:
        rendered_depth: (H, W) or (H, W, 1)
        mono_depth: (H, W)
        mask: optional (H, W) boolean mask
    """
    rd = rendered_depth.squeeze()
    md = mono_depth.squeeze()

    if mask is not None:
        rd = rd[mask]
        md = md[mask]
    else:
        rd = rd.reshape(-1)
        md = md.reshape(-1)

    # Pearson correlation
    rd_centered = rd - rd.mean()
    md_centered = md - md.mean()
    corr = (rd_centered * md_centered).sum() / (
        rd_centered.norm() * md_centered.norm() + 1e-8
    )

    return 1.0 - corr  # loss = 1 - correlation


def reconstruction_loss(rendered: torch.Tensor, target: torch.Tensor,
                         ssim_fn: SSIMLoss, lpips_fn: LPIPSLoss | None = None,
                         cfg: RI3DConfig | None = None,
                         mask: torch.Tensor | None = None) -> torch.Tensor:
    """Combined reconstruction loss: L1 + SSIM + LPIPS.

    Args:
        rendered, target: (H, W, 3) images in [0, 1]
        mask: optional (H, W, 1) or (H, W) mask to apply
    """
    if cfg is None:
        cfg = RI3DConfig()

    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        rendered = rendered * mask
        target = target * mask

    l1 = F.l1_loss(rendered, target)
    ssim = ssim_fn(rendered, target)
    loss = cfg.loss_l1_weight * l1 + cfg.loss_ssim_weight * ssim

    if lpips_fn is not None:
        lpips_val = lpips_fn(rendered, target)
        loss = loss + cfg.loss_lpips_weight * lpips_val

    return loss


class PlateauDetector:
    """Detects loss plateau for early stopping."""

    def __init__(self, window: int = 200, threshold: float = 1e-4, min_iters: int = 1000):
        self.window = window
        self.threshold = threshold
        self.min_iters = min_iters
        self.losses = []

    def update(self, loss: float) -> bool:
        """Add loss value. Returns True if plateau detected."""
        self.losses.append(loss)
        n = len(self.losses)

        if n < self.min_iters or n < self.window * 2:
            return False

        # Compare mean of recent window vs previous window
        recent = sum(self.losses[-self.window:]) / self.window
        previous = sum(self.losses[-2*self.window:-self.window]) / self.window

        if previous == 0:
            return False

        improvement = (previous - recent) / abs(previous)
        return improvement < self.threshold
