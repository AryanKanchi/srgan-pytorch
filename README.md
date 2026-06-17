<div align="center">

# SRGAN — Super-Resolution GAN

**PyTorch implementation of Photo-Realistic Single Image Super-Resolution**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-CVPR%202017-red)](https://arxiv.org/abs/1609.04802)

*Based on Ledig et al., "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network", CVPR 2017*

</div>

---

## What is SRGAN?

SRGAN recovers a **4× high-resolution image** from a single low-resolution input using a GAN trained with **perceptual loss** (VGG19 feature space) instead of pixel-wise MSE. This produces sharp, photorealistic textures that classical interpolation methods cannot recover.

| | Bicubic | SRGAN (ours) | SRGAN (paper) |
|---|---|---|---|
| **PSNR (dB)** | 26.50 | 28.45 | 29.40 |
| **SSIM** | 0.742 | 0.835 | 0.847 |

> Trained on DIV2K (800 images), 8 residual blocks, 8 pre-train + 20 adversarial epochs on a T4 GPU.

---

## Results

> _Sample comparison grids (LR bicubic ↑ | SRGAN output | HR ground truth)_

| LR (Bicubic ×4) | SRGAN (ours) | HR Ground Truth |
|:---:|:---:|:---:|
| ![](results/samples/bicubic.png) | ![](results/samples/srgan.png) | ![](results/samples/hr.png) |

> Loss curves and PSNR/SSIM progression saved to `results/training_curves.png` after training.

---

## Architecture

```
Generator (SRResNet)
───────────────────────────────────────────────────────
Input (3×24×24)
  → Conv(9×9, 64) + PReLU                [head]
  → 16× ResidualBlock                    [Conv-BN-PReLU-Conv-BN + skip]
  → Conv(3×3, 64) + BN + long skip       [post-residual]
  → UpsampleBlock ×2                     [Conv + PixelShuffle(×2) + PReLU]
  → Conv(9×9, 3) + Tanh
Output (3×96×96)

Discriminator (VGG-style)
───────────────────────────────────────────────────────
Input (3×96×96)
  → 8× ConvBlock (alternating stride 1/2, LeakyReLU)
  → Flatten → Linear(1024) → LeakyReLU → Linear(1) → Sigmoid

Loss
───────────────────────────────────────────────────────
L_G = L_perceptual (VGG relu2_2) + 1e-3 × L_adversarial (BCE)
L_D = BCE(real) + BCE(fake)
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/srgan.git
cd srgan
pip install -r requirements.txt
```

### 2. Train on STL-10 (auto-downloads)

```bash
python srgan_complete.py --train
```

### 3. Train on your own images (e.g. DIV2K)

```bash
python srgan_complete.py --train --custom --data /path/to/DIV2K/
```

### 4. Inference on a single image

```bash
python srgan_complete.py --infer --image myimage.jpg --output result.png
```

---

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--train` | — | Run full two-phase training |
| `--infer` | — | Run inference on a single image |
| `--image` | — | Path to input image (inference mode) |
| `--output` | `sr_output.png` | Output path (inference mode) |
| `--checkpoint` | `checkpoints/best_model.pth` | Model checkpoint |
| `--data` | `data/` | Dataset directory |
| `--custom` | `False` | Use custom image folder instead of STL-10 |
| `--epochs` | `100` | Total GAN training epochs |
| `--batch` | `16` | Batch size |

---

## Project Structure

```
srgan/
├── srgan_complete.py       ← Full implementation (models, training, inference)
├── requirements.txt        ← Python dependencies
├── checkpoints/            ← Saved weights (auto-created)
│   ├── best_model.pth      ← Best checkpoint by PSNR
│   └── pretrain_generator.pth
├── results/                ← Comparison grids + training curves (auto-created)
│   └── training_curves.png
└── logs/
    └── metrics.json        ← Full training history (loss, PSNR, SSIM per epoch)
```

---

## Training Details

**Phase 1 — SRResNet pre-training (MSE only)**
Generator trained in isolation with pixel-wise MSE for 20 epochs. This avoids mode collapse when adversarial loss is introduced cold.

**Phase 2 — Adversarial training**
Generator and discriminator trained jointly for 100 epochs. LR decays at 50% and 75% of training via MultiStepLR.

| Hyperparameter | Value |
|---|---|
| Scale factor | ×4 (24→96 px) |
| Residual blocks | 16 (configurable) |
| Batch size | 16 |
| Generator LR | 1e-4 |
| Discriminator LR | 1e-4 |
| VGG loss layer | relu2_2 |
| Adversarial λ | 1e-3 |
| Optimizer | Adam (β₁=0.9, β₂=0.999) |

---

## Authors

**Aryan K A** · **Botta Vighnesh**
B.E. AI & ML, B.M.S. College of Engineering, Bengaluru
AAT for course 24AM5AEGAN — Autoencoder and Generative AI

---

## References

- Ledig, C., et al. (2017). [Photo-realistic single image super-resolution using a generative adversarial network](https://arxiv.org/abs/1609.04802). *CVPR 2017*.
- Shi, W., et al. (2016). [Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network](https://arxiv.org/abs/1609.05158). *CVPR 2016*.
- Wang, X., et al. (2018). [ESRGAN: Enhanced super-resolution generative adversarial networks](https://arxiv.org/abs/1809.00219). *ECCV Workshops*.
