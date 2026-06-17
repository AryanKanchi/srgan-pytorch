"""
============================================================
  SRGAN — Super-Resolution Generative Adversarial Network
  Implementation in PyTorch
  Based on: Ledig et al., "Photo-Realistic Single Image
  Super-Resolution Using a Generative Adversarial Network"
  CVPR 2017  (arXiv:1609.04802)
============================================================
  Authors : Aryan K A | Botta Vighnesh (BMSCE, AI & ML Dept.)
  Dataset : STL-10 (auto-downloaded) or custom images
  Scale   : ×4  (e.g. 24×24  →  96×96)
  Losses  : Perceptual (VGG19) + Adversarial (BCE)
============================================================
"""

# ─── Standard Library ─────────────────────────────────────
import os
import math
import time
import json
import argparse
from pathlib import Path

# ─── Third-party ──────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import make_grid, save_image

# ─── Reproducibility ──────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════
#  SECTION 1 – HYPER-PARAMETERS
# ══════════════════════════════════════════════════════════

class Config:
    # Image
    HR_SIZE      = 96          # High-resolution patch size
    LR_SIZE      = 24          # Low-resolution patch size  (HR / SCALE)
    SCALE        = 4           # Upscaling factor

    # Architecture
    NUM_RES_BLOCKS = 16        # Residual blocks in generator
    NUM_FILTERS    = 64        # Base feature channels

    # Training
    BATCH_SIZE   = 16
    NUM_EPOCHS   = 100         # Full training epochs
    PRETRAIN_EPOCHS = 20       # SRResNet pre-training (MSE only)
    LR_G         = 1e-4        # Generator learning rate
    LR_D         = 1e-4        # Discriminator learning rate
    BETA1        = 0.9         # Adam β₁
    BETA2        = 0.999       # Adam β₂

    # Loss weights
    LAMBDA_PERCEPT = 1.0       # Perceptual loss weight
    LAMBDA_ADV     = 1e-3      # Adversarial loss weight

    # VGG layer for perceptual loss
    VGG_LAYER = "relu2_2"      # Feature layer name

    # Paths
    CHECKPOINT_DIR = Path("checkpoints")
    RESULTS_DIR    = Path("results")
    LOG_DIR        = Path("logs")

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluation
    EVAL_EVERY = 10            # Evaluate every N epochs

CFG = Config()


# ══════════════════════════════════════════════════════════
#  SECTION 2 – DATASET
# ══════════════════════════════════════════════════════════

class SRDataset(Dataset):
    """
    Super-resolution dataset.

    Loads HR images, applies random crop to CFG.HR_SIZE,
    then produces the LR counterpart by bicubic downsampling
    to CFG.LR_SIZE.  Both are normalised to [-1, 1].

    If `root` contains a sub-folder of images use `custom=True`;
    otherwise STL-10 (96×96) is downloaded automatically —
    perfect for SRGAN (already 96 px, no extra crop needed).
    """

    def __init__(self, root: str = "data", split: str = "train",
                 custom: bool = False):
        self.custom = custom

        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(CFG.HR_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

        self.lr_transform = transforms.Compose([
            transforms.Resize(CFG.LR_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

        if custom:
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            self.image_paths = [
                p for p in Path(root).rglob("*") if p.suffix.lower() in exts
            ]
            if not self.image_paths:
                raise FileNotFoundError(f"No images found under {root}")
        else:
            self.dataset = torchvision.datasets.STL10(
                root=root, split="unlabeled" if split == "train" else "test",
                download=True,
                transform=None,
            )

    def __len__(self) -> int:
        return len(self.image_paths) if self.custom else len(self.dataset)

    def __getitem__(self, idx: int):
        if self.custom:
            img = Image.open(self.image_paths[idx]).convert("RGB")
        else:
            img, _ = self.dataset[idx]
            # torchvision STL-10 already returns a PIL Image when transform=None
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)

        # HR crop
        hr_tensor = self.hr_transform(img)

        # Rebuild PIL from HR tensor (un-normalize) then downscale
        hr_pil = transforms.ToPILImage()(
            (hr_tensor * 0.5 + 0.5).clamp(0, 1)
        )
        lr_tensor = self.lr_transform(hr_pil)

        return lr_tensor, hr_tensor


# ══════════════════════════════════════════════════════════
#  SECTION 3 – GENERATOR (SRResNet)
# ══════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """
    SRGAN residual block:
        Conv → BN → PReLU → Conv → BN
    with a skip connection.  No activation after the addition
    (following the original paper).
    """
    def __init__(self, channels: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.8),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    """
    Sub-pixel convolution block (Shi et al., 2016).
    Conv → PixelShuffle(×2) → PReLU
    Two of these in sequence give ×4 upscaling.
    """
    def __init__(self, channels: int, scale_factor: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * (scale_factor ** 2),
                      kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(scale_factor),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Generator(nn.Module):
    """
    SRResNet / SRGAN Generator.

    Architecture:
        Conv-PReLU                          (initial feature extraction)
        → N × ResidualBlock                 (deep residual learning)
        → Conv-BN                           (post-residual conv)
        → ×2 UpsampleBlock × 2             (×4 total upscaling via PixelShuffle)
        → Conv                              (final RGB reconstruction)
        → Tanh                              (output in [-1, 1])

    Parameters
    ----------
    scale_factor : int
        Total spatial upscaling. Must be a power of 2.
    num_res_blocks : int
        Number of residual blocks (paper uses 16).
    num_filters : int
        Base channel count (paper uses 64).
    """

    def __init__(self, scale_factor: int = 4,
                 num_res_blocks: int = 16,
                 num_filters: int = 64):
        super().__init__()
        self.scale_factor = scale_factor

        # ── Head ────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=9, padding=4, bias=False),
            nn.PReLU(),
        )

        # ── Residual body ────────────────────────────────
        res_blocks = [ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        self.body = nn.Sequential(*res_blocks)

        # ── Post-residual conv ───────────────────────────
        self.post_res = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters, momentum=0.8),
        )

        # ── Upsampling ───────────────────────────────────
        up_blocks = []
        num_up = int(math.log2(scale_factor))  # 4 → 2 blocks of ×2
        for _ in range(num_up):
            up_blocks.append(UpsampleBlock(num_filters, scale_factor=2))
        self.upsample = nn.Sequential(*up_blocks)

        # ── Tail ─────────────────────────────────────────
        self.tail = nn.Sequential(
            nn.Conv2d(num_filters, 3, kernel_size=9, padding=4, bias=False),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_out  = self.head(x)
        body_out  = self.body(head_out)
        res_out   = self.post_res(body_out) + head_out   # long skip connection
        up_out    = self.upsample(res_out)
        return self.tail(up_out)


# ══════════════════════════════════════════════════════════
#  SECTION 4 – DISCRIMINATOR
# ══════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """
    Discriminator conv block: Conv → (BN) → LeakyReLU
    BatchNorm omitted on the first block.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, use_bn: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                      padding=1, bias=not use_bn)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Discriminator(nn.Module):
    """
    SRGAN VGG-style discriminator.

    8 convolutional blocks with alternating stride-1 / stride-2,
    followed by two dense layers and sigmoid output.

    Input:  (B, 3, 96, 96) HR or SR image
    Output: (B, 1) real/fake probability
    """

    def __init__(self, hr_size: int = 96, num_filters: int = 64):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 — no BN
            ConvBlock(3,              num_filters,    stride=1, use_bn=False),
            ConvBlock(num_filters,    num_filters,    stride=2, use_bn=True),
            # Block 2
            ConvBlock(num_filters,    num_filters*2,  stride=1),
            ConvBlock(num_filters*2,  num_filters*2,  stride=2),
            # Block 3
            ConvBlock(num_filters*2,  num_filters*4,  stride=1),
            ConvBlock(num_filters*4,  num_filters*4,  stride=2),
            # Block 4
            ConvBlock(num_filters*4,  num_filters*8,  stride=1),
            ConvBlock(num_filters*8,  num_filters*8,  stride=2),
        )

        # Spatial size after 4× stride-2 convolutions on 96px input: 6×6
        flat_size = num_filters * 8 * (hr_size // 16) ** 2

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat)


# ══════════════════════════════════════════════════════════
#  SECTION 5 – PERCEPTUAL (VGG) LOSS
# ══════════════════════════════════════════════════════════

class VGGPerceptualLoss(nn.Module):
    """
    Content (perceptual) loss using pre-trained VGG19 features.

    Extracts intermediate feature maps from both the SR and HR
    images and minimises their MSE.  Feature statistics in
    perceptual space capture texture and structure better than
    pixel-wise MSE alone.

    Uses relu2_2 features by default (VGG19 block2_relu2).
    The VGG19 weights are frozen — no gradients flow through them.
    """

    # Map readable names to VGG19 layer indices
    LAYER_MAP = {
        "relu1_2": 4,
        "relu2_2": 9,   # ← paper's choice
        "relu3_4": 18,
        "relu4_4": 27,
    }

    def __init__(self, layer: str = "relu2_2"):
        super().__init__()
        if layer not in self.LAYER_MAP:
            raise ValueError(f"Unknown VGG layer '{layer}'. "
                             f"Choose from {list(self.LAYER_MAP)}")

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        cut = self.LAYER_MAP[layer]
        self.feature_extractor = nn.Sequential(
            *list(vgg.features.children())[:cut + 1]
        )
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        # ImageNet normalisation (expected by VGG)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.loss_fn = nn.MSELoss()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from [-1,1] to ImageNet-normalised space."""
        x = (x + 1.0) / 2.0          # → [0, 1]
        return (x - self.mean) / self.std

    def forward(self, sr: torch.Tensor,
                hr: torch.Tensor) -> torch.Tensor:
        sr_feat = self.feature_extractor(self._normalize(sr))
        with torch.no_grad():
            hr_feat = self.feature_extractor(self._normalize(hr))
        return self.loss_fn(sr_feat, hr_feat.detach())


# ══════════════════════════════════════════════════════════
#  SECTION 6 – EVALUATION METRICS
# ══════════════════════════════════════════════════════════

def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(B,3,H,W) in [-1,1] → (B,H,W,3) uint8."""
    arr = ((t.detach().cpu() + 1.0) / 2.0 * 255.0).clamp(0, 255)
    return arr.permute(0, 2, 3, 1).numpy().astype(np.uint8)


def compute_psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).
    PSNR = 10 × log₁₀( MAX² / MSE )
    Higher is better; >30 dB is generally considered good.
    """
    mse = F.mse_loss(sr, hr).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(4.0 / mse)   # MAX=2 because range is [-1,1] → MAX²=4


def compute_ssim(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """
    Structural Similarity Index (SSIM) — averaged over batch.
    Uses a simplified version without a Gaussian window for speed.
    Range: [-1, 1], closer to 1 is better.
    """
    sr_arr = tensor_to_uint8(sr).astype(np.float64)
    hr_arr = tensor_to_uint8(hr).astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    ssim_vals = []
    for s, h in zip(sr_arr, hr_arr):
        mu1, mu2 = s.mean(), h.mean()
        sigma1 = s.std()
        sigma2 = h.std()
        sigma12 = np.mean((s - mu1) * (h - mu2))
        ssim = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2))
        ssim_vals.append(ssim)
    return float(np.mean(ssim_vals))


# ══════════════════════════════════════════════════════════
#  SECTION 7 – UTILITY: LOGGING & VISUALISATION
# ══════════════════════════════════════════════════════════

class TrainingLogger:
    """Lightweight CSV + JSON logger for metrics."""

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / "metrics.json"
        self.history: dict = {
            "epoch": [], "loss_g": [], "loss_d": [],
            "psnr": [], "ssim": [], "time_s": [],
        }

    def log(self, epoch: int, loss_g: float, loss_d: float,
            psnr: float = 0.0, ssim: float = 0.0, elapsed: float = 0.0):
        self.history["epoch"].append(epoch)
        self.history["loss_g"].append(round(loss_g, 6))
        self.history["loss_d"].append(round(loss_d, 6))
        self.history["psnr"].append(round(psnr, 4))
        self.history["ssim"].append(round(ssim, 4))
        self.history["time_s"].append(round(elapsed, 2))

        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"  Epoch {epoch:4d} | "
              f"G: {loss_g:.4f}  D: {loss_d:.4f} | "
              f"PSNR: {psnr:6.2f} dB  SSIM: {ssim:.4f} | "
              f"Time: {elapsed:.1f}s")


def save_comparison_grid(lr: torch.Tensor, sr: torch.Tensor,
                         hr: torch.Tensor, path: Path,
                         max_images: int = 4):
    """
    Saves a side-by-side grid: LR (bicubic ×4) | SR | HR
    """
    n = min(max_images, lr.size(0))

    # Bicubic upsample LR for fair visual comparison
    lr_up = F.interpolate(lr[:n], size=(CFG.HR_SIZE, CFG.HR_SIZE),
                          mode="bicubic", align_corners=False)

    row = torch.cat([lr_up, sr[:n], hr[:n]], dim=0)
    grid = make_grid(row * 0.5 + 0.5, nrow=n, padding=2, normalize=False)
    save_image(grid, path)


def plot_training_curves(logger: TrainingLogger, save_dir: Path):
    """Plots loss and metric curves and saves to PNG."""
    h = logger.history
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curves
    axes[0].plot(h["epoch"], h["loss_g"], label="Generator loss", color="#7F77DD")
    axes[0].plot(h["epoch"], h["loss_d"], label="Discriminator loss", color="#D85A30")
    axes[0].set_title("Training losses"); axes[0].legend()
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")

    # PSNR
    axes[1].plot(h["epoch"], h["psnr"], color="#1D9E75")
    axes[1].set_title("PSNR over epochs")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("PSNR (dB)")

    # SSIM
    axes[2].plot(h["epoch"], h["ssim"], color="#185FA5")
    axes[2].set_title("SSIM over epochs")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("SSIM")

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close()
    print(f"  ✔ Training curves saved to {save_dir / 'training_curves.png'}")


# ══════════════════════════════════════════════════════════
#  SECTION 8 – TRAINING LOOP
# ══════════════════════════════════════════════════════════

def pretrain_generator(generator: Generator,
                       dataloader: DataLoader,
                       logger: TrainingLogger) -> Generator:
    """
    Phase 1 — SRResNet pre-training with pixel-wise MSE loss.
    This stabilises the generator before adversarial training.
    Following the paper, we train for CFG.PRETRAIN_EPOCHS epochs.
    """
    print("\n" + "="*55)
    print("  Phase 1 — SRResNet pre-training (MSE only)")
    print("="*55)

    optimizer = optim.Adam(generator.parameters(),
                           lr=CFG.LR_G, betas=(CFG.BETA1, CFG.BETA2))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    mse_loss  = nn.MSELoss()

    generator.train()
    for epoch in range(1, CFG.PRETRAIN_EPOCHS + 1):
        epoch_loss = 0.0
        t0 = time.time()
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs = lr_imgs.to(CFG.DEVICE)
            hr_imgs = hr_imgs.to(CFG.DEVICE)

            sr_imgs = generator(lr_imgs)
            loss    = mse_loss(sr_imgs, hr_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        elapsed  = time.time() - t0

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [Pre-train] Epoch {epoch:3d}/{CFG.PRETRAIN_EPOCHS}  "
                  f"MSE: {avg_loss:.4f}  ({elapsed:.1f}s)")

    print("  ✔ Pre-training complete.\n")
    return generator


def train_srgan(generator: Generator,
                discriminator: Discriminator,
                dataloader: DataLoader,
                val_dataloader: DataLoader,
                logger: TrainingLogger) -> dict:
    """
    Phase 2 — Full adversarial SRGAN training.

    Generator loss:
        L_G = L_perceptual + λ_adv × L_adversarial

    Discriminator loss:
        L_D = -[log D(HR) + log(1 - D(G(LR)))]
    """
    print("="*55)
    print("  Phase 2 — SRGAN adversarial training")
    print("="*55)

    # ── Losses ──────────────────────────────────────────
    perceptual_loss = VGGPerceptualLoss(layer=CFG.VGG_LAYER).to(CFG.DEVICE)
    adversarial_loss = nn.BCELoss()
    mse_loss         = nn.MSELoss()

    # ── Optimisers ──────────────────────────────────────
    opt_G = optim.Adam(generator.parameters(),
                       lr=CFG.LR_G, betas=(CFG.BETA1, CFG.BETA2))
    opt_D = optim.Adam(discriminator.parameters(),
                       lr=CFG.LR_D, betas=(CFG.BETA1, CFG.BETA2))

    # Decay LR by ×0.1 at 50% and 75% of training
    sch_G = optim.lr_scheduler.MultiStepLR(
        opt_G, milestones=[CFG.NUM_EPOCHS//2, 3*CFG.NUM_EPOCHS//4], gamma=0.1)
    sch_D = optim.lr_scheduler.MultiStepLR(
        opt_D, milestones=[CFG.NUM_EPOCHS//2, 3*CFG.NUM_EPOCHS//4], gamma=0.1)

    # ── Fixed val batch for consistent comparison grids ─
    fixed_lr, fixed_hr = next(iter(val_dataloader))
    fixed_lr = fixed_lr.to(CFG.DEVICE)
    fixed_hr = fixed_hr.to(CFG.DEVICE)

    best_psnr = 0.0

    for epoch in range(1, CFG.NUM_EPOCHS + 1):
        generator.train()
        discriminator.train()
        t0 = time.time()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for lr_imgs, hr_imgs in dataloader:
            lr_imgs = lr_imgs.to(CFG.DEVICE)
            hr_imgs = hr_imgs.to(CFG.DEVICE)
            batch   = lr_imgs.size(0)

            real_labels = torch.ones(batch, 1, device=CFG.DEVICE)
            fake_labels = torch.zeros(batch, 1, device=CFG.DEVICE)

            # ── Train Discriminator ──────────────────────
            discriminator.zero_grad()
            sr_imgs = generator(lr_imgs).detach()  # no G grad here

            d_real = discriminator(hr_imgs)
            d_fake = discriminator(sr_imgs)

            loss_d_real = adversarial_loss(d_real, real_labels)
            loss_d_fake = adversarial_loss(d_fake, fake_labels)
            loss_d = (loss_d_real + loss_d_fake) / 2.0

            loss_d.backward()
            opt_D.step()

            # ── Train Generator ──────────────────────────
            generator.zero_grad()
            sr_imgs   = generator(lr_imgs)          # fresh pass WITH grad
            d_sr      = discriminator(sr_imgs)

            # Perceptual content loss (VGG feature space)
            l_percept = perceptual_loss(sr_imgs, hr_imgs)

            # Adversarial loss — fool discriminator
            l_adv = adversarial_loss(d_sr, real_labels)

            # Total generator loss
            loss_g = (CFG.LAMBDA_PERCEPT * l_percept
                      + CFG.LAMBDA_ADV   * l_adv)

            loss_g.backward()
            opt_G.step()

            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()

        sch_G.step()
        sch_D.step()

        avg_g = epoch_g_loss / len(dataloader)
        avg_d = epoch_d_loss / len(dataloader)
        elapsed = time.time() - t0

        # ── Evaluation ──────────────────────────────────
        psnr_val = ssim_val = 0.0
        if epoch % CFG.EVAL_EVERY == 0 or epoch == CFG.NUM_EPOCHS:
            generator.eval()
            with torch.no_grad():
                psnr_list, ssim_list = [], []
                for lr_b, hr_b in val_dataloader:
                    lr_b = lr_b.to(CFG.DEVICE)
                    hr_b = hr_b.to(CFG.DEVICE)
                    sr_b = generator(lr_b)
                    psnr_list.append(compute_psnr(sr_b, hr_b))
                    ssim_list.append(compute_ssim(sr_b, hr_b))
                psnr_val = float(np.mean(psnr_list))
                ssim_val = float(np.mean(ssim_list))

                # Save comparison grid
                sr_fixed = generator(fixed_lr)
                grid_path = CFG.RESULTS_DIR / f"epoch_{epoch:04d}.png"
                save_comparison_grid(fixed_lr, sr_fixed, fixed_hr, grid_path)

            # Save best checkpoint
            if psnr_val > best_psnr:
                best_psnr = psnr_val
                torch.save({
                    "epoch": epoch,
                    "generator_state": generator.state_dict(),
                    "discriminator_state": discriminator.state_dict(),
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                }, CFG.CHECKPOINT_DIR / "best_model.pth")
                print(f"  ★ New best PSNR: {best_psnr:.2f} dB — checkpoint saved")

        logger.log(epoch, avg_g, avg_d, psnr_val, ssim_val, elapsed)

    return logger.history


# ══════════════════════════════════════════════════════════
#  SECTION 9 – INFERENCE
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def super_resolve_image(image_path: str,
                        checkpoint_path: str,
                        output_path: str = "sr_output.png"):
    """
    Inference function.
    Loads a single image, downscales it by SCALE to simulate LR,
    then super-resolves and saves the result alongside the bicubic
    baseline and original.

    Usage:
        python srgan_complete.py --infer --image myimage.jpg
    """
    generator = Generator(
        scale_factor=CFG.SCALE,
        num_res_blocks=CFG.NUM_RES_BLOCKS,
        num_filters=CFG.NUM_FILTERS,
    ).to(CFG.DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=CFG.DEVICE)
    generator.load_state_dict(ckpt["generator_state"])
    generator.eval()

    img = Image.open(image_path).convert("RGB")
    # Ensure divisible by scale
    w, h = img.size
    w = (w // CFG.SCALE) * CFG.SCALE
    h = (h // CFG.SCALE) * CFG.SCALE
    img = img.crop((0, 0, w, h))

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    hr_t = to_tensor(img).unsqueeze(0).to(CFG.DEVICE)
    lr_t = F.interpolate(hr_t, scale_factor=1/CFG.SCALE,
                         mode="bicubic", align_corners=False)
    sr_t = generator(lr_t)

    # Bicubic upsampled baseline
    bic_t = F.interpolate(lr_t, scale_factor=CFG.SCALE,
                          mode="bicubic", align_corners=False)

    # Build comparison strip: Bicubic | SRGAN | Original
    comparison = torch.cat([bic_t, sr_t, hr_t], dim=3)  # concat width
    save_image(comparison * 0.5 + 0.5, output_path)
    print(f"  Saved: Bicubic | SRGAN | Original → {output_path}")

    # Print PSNR comparison
    psnr_bic = compute_psnr(bic_t, hr_t)
    psnr_sr  = compute_psnr(sr_t, hr_t)
    print(f"  Bicubic PSNR: {psnr_bic:.2f} dB")
    print(f"  SRGAN   PSNR: {psnr_sr:.2f} dB")
    print(f"  Improvement : {psnr_sr - psnr_bic:+.2f} dB")


# ══════════════════════════════════════════════════════════
#  SECTION 10 – MAIN
# ══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="SRGAN — PyTorch Implementation")
    p.add_argument("--train",       action="store_true", help="Run training")
    p.add_argument("--infer",       action="store_true", help="Run inference")
    p.add_argument("--image",       type=str, default=None,
                   help="Image path for inference")
    p.add_argument("--checkpoint",  type=str,
                   default=str(CFG.CHECKPOINT_DIR / "best_model.pth"),
                   help="Checkpoint path for inference")
    p.add_argument("--data",        type=str, default="data",
                   help="Dataset root directory")
    p.add_argument("--custom",      action="store_true",
                   help="Use custom image folder instead of STL-10")
    p.add_argument("--epochs",      type=int, default=CFG.NUM_EPOCHS)
    p.add_argument("--batch",       type=int, default=CFG.BATCH_SIZE)
    p.add_argument("--output",      type=str, default="sr_output.png")
    return p.parse_args()


def main():
    args = parse_args()

    # Override config from CLI
    CFG.NUM_EPOCHS  = args.epochs
    CFG.BATCH_SIZE  = args.batch

    # Create directories
    for d in [CFG.CHECKPOINT_DIR, CFG.RESULTS_DIR, CFG.LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  SRGAN — Super-Resolution GAN")
    print(f"  Device  : {CFG.DEVICE}")
    print(f"  Scale   : ×{CFG.SCALE}  ({CFG.LR_SIZE}→{CFG.HR_SIZE})")
    print(f"  Epochs  : {CFG.NUM_EPOCHS}  (pre-train: {CFG.PRETRAIN_EPOCHS})")
    print(f"  Batch   : {CFG.BATCH_SIZE}")
    print(f"{'='*55}\n")

    # ── Inference mode ───────────────────────────────────
    if args.infer:
        if not args.image:
            raise ValueError("Provide --image path for inference mode.")
        super_resolve_image(args.image, args.checkpoint, args.output)
        return

    # ── Training mode ────────────────────────────────────
    if not args.train:
        print("Tip: pass --train to start training, or --infer to run inference.")
        print("Running a quick sanity check instead...\n")

    print("  Loading dataset...")
    train_ds = SRDataset(root=args.data, split="train",  custom=args.custom)
    val_ds   = SRDataset(root=args.data, split="test",   custom=args.custom)

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE,
                              shuffle=True,  num_workers=2,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG.BATCH_SIZE,
                              shuffle=False, num_workers=2,
                              pin_memory=True)

    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val samples   : {len(val_ds)}")

    # ── Build models ─────────────────────────────────────
    generator     = Generator(
        scale_factor    = CFG.SCALE,
        num_res_blocks  = CFG.NUM_RES_BLOCKS,
        num_filters     = CFG.NUM_FILTERS,
    ).to(CFG.DEVICE)

    discriminator = Discriminator(
        hr_size     = CFG.HR_SIZE,
        num_filters = CFG.NUM_FILTERS,
    ).to(CFG.DEVICE)

    # Parameter counts
    g_params = sum(p.numel() for p in generator.parameters()     if p.requires_grad)
    d_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"\n  Generator params     : {g_params:,}")
    print(f"  Discriminator params : {d_params:,}\n")

    logger = TrainingLogger(CFG.LOG_DIR)

    # ── Phase 1: Pre-train generator ─────────────────────
    generator = pretrain_generator(generator, train_loader, logger)

    # Save pre-trained checkpoint
    torch.save({
        "generator_state": generator.state_dict(),
        "phase": "pretrain",
    }, CFG.CHECKPOINT_DIR / "pretrain_generator.pth")

    # ── Phase 2: Adversarial training ────────────────────
    history = train_srgan(generator, discriminator,
                          train_loader, val_loader, logger)

    # ── Plot training curves ──────────────────────────────
    plot_training_curves(logger, CFG.RESULTS_DIR)

    # ── Final summary ────────────────────────────────────
    valid_psnr = [p for p in history["psnr"] if p > 0]
    if valid_psnr:
        print(f"\n  Final best PSNR : {max(valid_psnr):.2f} dB")
        print(f"  Final best SSIM : {max(history['ssim']):.4f}")
    print(f"\n  Results → {CFG.RESULTS_DIR}/")
    print(f"  Checkpoints → {CFG.CHECKPOINT_DIR}/")
    print("\n  Training complete!\n")


if __name__ == "__main__":
    main()
