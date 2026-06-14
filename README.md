# Plant Leaves Super-Resolution using Conditional GAN

**Author:** Hasrat Hussain | **Roll Number:** 24f1002299  
**Platform:** Kaggle (Plant Leaves Super-Resolution Challenge)  
**Notebook:** `24f1002299-notebook-26T1-nppe2`

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Architecture](#architecture)
   - [Generator: RRDB-Net with Channel Attention](#generator-rrdb-net-with-channel-attention)
   - [Discriminator: U-Net PatchGAN](#discriminator-u-net-patchgan)
4. [Loss Functions](#loss-functions)
5. [Training Strategy](#training-strategy)
   - [Phase 1: Pixel Pretraining](#phase-1-pixel-pretraining-350-epochs)
   - [Phase 2: GAN Fine-tuning](#phase-2-gan-fine-tuning-50-epochs)
6. [Inference Techniques](#inference-techniques)
7. [Results](#results)
8. [Key Design Decisions](#key-design-decisions)
9. [Environment and Dependencies](#environment-and-dependencies)
10. [Repository Structure](#repository-structure)
11. [References](#references)

---

## Project Overview

This project addresses the Kaggle **Plant Leaves Super-Resolution Challenge**, which requires reconstructing high-resolution (128x128) plant leaf images from degraded low-resolution (32x32) inputs. The task is a 4x single-image super-resolution problem.

Plant leaf imagery is particularly challenging for super-resolution because leaf surfaces contain fine repeating vein structures, subtle disease patterns, color gradients, and edge details that are critical for downstream agricultural analysis. Simple interpolation methods such as bicubic resizing can upscale an image spatially, but they systematically fail to recover missing high-frequency information.

The solution employs a two-phase conditional GAN (cGAN) training pipeline built around an RRDB-Net generator (inspired by ESRGAN) combined with a U-Net discriminator. The final model achieves a full-pass MAE of **16.67** (on a 0-255 pixel scale) compared to the bicubic baseline of **18.40**, representing a relative improvement of approximately **9.4%**.

---

## Dataset

The dataset was provided by the Kaggle competition as part of the AgriVision collection.

| Split    | Count |
|----------|-------|
| Training LR images (32x32)   | 1,642 |
| Training HR images (128x128) | 1,642 |
| Test LR images (32x32)       | 495   |

All images are RGB PNG files. The LR and HR training sets are paired by filename. A notable characteristic of the dataset is a dynamic range asymmetry: LR pixel values span approximately [0.0, 0.710] in float32, while HR images span the full [0.0, 1.0]. This means the degradation process compresses the dynamic range in addition to reducing spatial resolution, making reconstruction harder than simple upscaling.

**Bicubic Baseline MAE:** 18.40 (on 0-255 scale), measured by resizing each training LR image with bicubic interpolation and comparing to the HR ground truth.

---

## Architecture

### Generator: RRDB-Net with Channel Attention

The generator is a custom RRDB-Net (Residual-in-Residual Dense Block Network) adapted from ESRGAN with the addition of Squeeze-and-Excitation (SE) channel attention.

**Overall data flow:**

```
Conv_first  -->  20 x RRDB(+SE)  -->  trunk_conv  -->  [Global Residual]
            -->  PixelShuffle(2x)  -->  PixelShuffle(2x)  -->  hr_conv  -->  Conv_last
```

**Building blocks:**

- **DenseLayer:** A single 3x3 convolution followed by LeakyReLU(0.2) and channel concatenation, producing a growth of 32 channels per layer.
- **ResidualDenseBlock (RDB):** Five stacked dense layers reducing back to 64 channels via a final 3x3 convolution. Followed by SE channel attention with reduction ratio 16. Residual scaling of beta=0.2 is applied before the skip connection.
- **RRDB:** Three stacked RDBs with an outer residual skip scaled by beta=0.2. This double-residual structure allows very deep information flow without exploding gradients.
- **Upsampling:** Two sequential PixelShuffle stages, each performing a 2x spatial upscale, for a combined 4x factor. Both upsampling convolutions are initialized with ICNR (Integer Convolution Nearest-Neighbour Resize) to eliminate checkerboard artifacts from initialization.

**Model scale:**

| Component              | Parameters  |
|------------------------|-------------|
| Generator (total)      | 14,792,003  |
| Generator (trainable)  | 14,792,003  |

Weight initialization uses Kaiming Normal with LeakyReLU gain for all convolutional layers, and ICNR for the two PixelShuffle convolutions.

---

### Discriminator: U-Net PatchGAN

Rather than a standard PatchGAN (which outputs a single patch-level grid), the discriminator follows a U-Net architecture to provide dense per-pixel gradient feedback to the generator.

**Encoder** (strided convolutions with InstanceNorm):

```
128x128  -->  64x64  -->  32x32  -->  16x16  -->  8x8  (bottleneck)
```

**Decoder** (transposed convolutions with skip connections from each encoder stage):

```
8x8  -->  16x16  -->  32x32  -->  64x64  -->  128x128
```

The final output is a 128x128 map of per-pixel real/fake logits. This design provides spatially dense gradient signals back to the generator, encouraging it to reconstruct fine local details rather than just improving global statistics.

**Model scale:**

| Component               | Parameters |
|-------------------------|------------|
| Discriminator (total)   | 8,656,412  |

The discriminator was only introduced in Phase 2 and used with a strongly suppressed adversarial weight to avoid degrading the pixel accuracy achieved in Phase 1.

---

## Loss Functions

Four loss components are used, each targeting a different aspect of image quality:

### Charbonnier Loss

```
L_Charb(x, y) = sqrt((x - y)^2 + eps^2),   eps = 1e-6
```

A smooth approximation of L1 loss. Unlike standard L1, it has continuous gradients everywhere including near zero, which benefits fine-grained texture optimization in the later epochs of Phase 1.

### Multi-Scale Perceptual Loss (VGG-19)

Features are extracted from a frozen VGG-19 network at two depth levels:

- **Stage 1 (conv2_2, layers 0-8):** Low-level edges, color gradients, fine texture.
- **Stage 2 (conv4_4, layers 9-26):** High-level semantic structure and global context.

The perceptual loss is an L1 distance in feature space, weighted equally across both stages (w1=0.5, w2=0.5). The provided competition VGG-19 weights are used rather than downloading ImageNet pretrained weights.

### FFT Frequency Loss

```
L_FFT(x, y) = || |FFT(x)| - |FFT(y)| ||_1
```

A frequency-domain loss computed via `torch.fft.rfft2` with orthonormal normalization. Pixel losses and perceptual losses tend to under-penalize high-frequency reconstruction errors because they operate in spatial or deep feature space. The FFT loss directly penalizes deviations in the magnitude spectrum, explicitly encouraging the model to recover fine high-frequency leaf textures like veins and serrated edges.

### Adversarial Loss (Phase 2 only)

Standard Binary Cross-Entropy with logits applied to the U-Net discriminator output:

```
L_adv = -E[log D(G(x))]
```

Label smoothing was applied to discriminator training targets (real: 0.9, fake: 0.1) for training stability.

### Composite Generator Loss (Phase 2)

```
L_G = 1.0 * L_Charb  +  0.003 * L_Perc  +  0.1 * L_FFT  +  0.0002 * L_adv
```

The weights were deliberately designed to be conservative: the adversarial term (0.0002) is two orders of magnitude smaller than the pixel term (1.0), ensuring that GAN training cannot significantly degrade the pixel accuracy achieved in Phase 1.

---

## Training Strategy

Training was split into two sequential phases to avoid the instability of training a GAN from scratch.

### Phase 1: Pixel Pretraining (350 epochs)

The generator is trained alone using only Charbonnier loss. No discriminator is involved. The objective is to drive the generator to a high-quality pixel-accurate solution before introducing adversarial pressure.

| Hyperparameter         | Value                                                        |
|------------------------|--------------------------------------------------------------|
| Optimizer              | AdamW (beta1=0.9, beta2=0.999, weight decay=1e-4)           |
| Initial learning rate  | 2e-4                                                         |
| LR schedule            | CosineAnnealingWarmRestarts (T0=50, T_mult=2)               |
| Warm-restart cycles    | 3 full cycles (at epochs 50, 150, 350)                      |
| Loss                   | Charbonnier only                                             |
| Gradient clipping      | max_norm = 1.0                                               |
| Mixed precision        | AMP (autocast + GradScaler)                                  |
| EMA decay              | 0.9995 (maintained throughout)                               |
| Batch size             | 16                                                           |

The CosineAnnealingWarmRestarts schedule was configured so that 350 epochs align exactly with 3 complete cosine cycles (T0=50, cycle lengths: 50, 100, 200). Each restart raises the learning rate back to 2e-4, allowing the optimizer to escape local minima discovered in the previous cycle. This contributed substantially to the Phase 1 MAE improvement from 17.00 (epoch 150) to 16.63 (epoch 350).

### Phase 2: GAN Fine-tuning (50 epochs)

The discriminator is initialized and both networks are fine-tuned together using the composite loss. Phase 2 starts from the best Phase 1 checkpoint.

| Hyperparameter           | Value                                                    |
|--------------------------|----------------------------------------------------------|
| Generator LR             | 1e-5                                                     |
| Discriminator LR         | 1e-5                                                     |
| LR schedule (G and D)    | CosineAnnealingLR (T_max=50, eta_min=1e-7)              |
| Label smoothing          | real -> 0.9, fake -> 0.1                                |
| Adversarial weight       | 0.0002                                                   |
| Perceptual weight        | 0.003                                                    |
| FFT weight               | 0.1                                                      |
| Gradient clipping        | max_norm = 1.0 (applied to G and D separately)          |

Both the generator and discriminator use separate GradScaler instances for mixed precision stability during the GAN update.

---

## Inference Techniques

### Exponential Moving Average (EMA) Generator

An EMA shadow copy of the generator weights is maintained throughout both training phases with decay=0.9995. At each training step, EMA parameters are updated as:

```
EMA_params = decay * EMA_params + (1 - decay) * model_params
```

At inference, the EMA model produced lower MAE than the standard checkpoint (16.6675 vs 16.6717) and was used for all final predictions.

### Test-Time Augmentation (TTA)

Self-ensemble across 8 geometric variants is applied at inference:

- 4 rotations: 0, 90, 180, 270 degrees
- 2 horizontal flip states: original and flipped

For each test image, all 8 variants are independently passed through the EMA generator, the augmentation is geometrically reversed, and all 8 predictions are averaged pixel-wise. TTA requires no additional training and provides a consistent improvement by reducing model uncertainty at tile boundaries and rotationally ambiguous texture regions.

---

## Results

### Phase 1 Training Progression

| Epoch     | Charb Loss | MAE (0-255) | Best MAE | LR       |
|-----------|------------|-------------|----------|----------|
| 1         | 2.75837    | 90.98       | 90.98    | 2.00e-04 |
| 10        | 0.07070    | 18.02       | 18.02    | 1.81e-04 |
| 20        | 0.06897    | 17.58       | 17.58    | 1.31e-04 |
| 30        | 0.06841    | 17.44       | 17.44    | 6.92e-05 |
| 50        | 0.06785    | 17.30       | 17.30    | 2.00e-04 (restart) |
| 100       | 0.06735    | 17.17       | 17.15    | 1.00e-04 |
| 150       | 0.06677    | 17.03       | 17.00    | 2.00e-04 (restart) |
| 250       | 0.06619    | 16.88       | 16.87    | 1.00e-04 |
| 300       | 0.06555    | 16.71       | 16.70    | 2.94e-05 |
| 330       | 0.06527    | 16.64       | 16.63    | 4.99e-06 |
| 350       | 0.06527    | 16.64       | **16.63** | 2.00e-04 |

The model surpassed the bicubic baseline (18.40) by epoch 10 and continued improving through all three warm-restart cycles.

### Phase 2 Training Progression

| Epoch | G Loss  | D Loss  | MAE (0-255) | Best MAE |
|-------|---------|---------|-------------|----------|
| 1     | 0.07340 | 0.70020 | 16.68       | 16.68    |
| 10    | 0.07333 | 0.36807 | 16.72       | 16.66    |
| 20    | 0.07213 | 0.34061 | 16.68       | 16.66    |
| 30    | 0.07184 | 0.33616 | 16.66       | 16.66    |
| 50    | 0.07187 | 0.33421 | 16.68       | **16.65** |

The discriminator stabilized from an initial D loss of 0.700 to approximately 0.334 by epoch 30, indicating a stable adversarial equilibrium. The conservative adversarial weight kept the Phase 2 MAE within 0.02 of the Phase 1 best.

### Final Model Comparison

| Model / Method          | MAE (0-255)  |
|-------------------------|--------------|
| Bicubic interpolation   | 18.40        |
| Phase 1 generator (best checkpoint) | 16.63 |
| Phase 2 generator (best checkpoint) | 16.65 |
| Standard generator (full pass)      | 16.6717 |
| EMA generator (full pass)           | **16.6675** |

### Post-Training Per-Image Analysis (200 samples, no TTA)

| Metric      | Mean     | Std   | Min   | Max   |
|-------------|----------|-------|-------|-------|
| MAE (0-255) | 16.95    | 6.19  | 5.29  | 31.56 |
| PSNR (dB)   | 21.69 dB | 3.24  | --    | --    |

The large standard deviation in MAE (6.19) reflects variance across leaf types. The best per-image MAE of 5.29 corresponds to near-perfect reconstruction of low-frequency uniform leaf regions, while the worst case of 31.56 occurs on images with fine repeating vein structures where the degradation is most ambiguous.

---

## Key Design Decisions

**Charbonnier loss over standard L1:** Charbonnier produces continuous gradients everywhere including near zero, which stabilizes and accelerates convergence during the later epochs of Phase 1 when the loss values are very small.

**CosineAnnealingWarmRestarts with 3 full cycles:** Warm restarts allow the optimizer to escape local minima by periodically raising the learning rate. Aligning 350 epochs with exactly three complete cycles (50 + 100 + 200) was deliberate; the third and longest cycle (200 epochs) produced the majority of the gains from MAE 17.00 to 16.63.

**FFT frequency loss:** Leaf textures contain high-frequency components (veins, serrated edges) that pixel loss and perceptual loss both tend to blur. The FFT loss directly penalizes magnitude spectrum differences and was critical for recovering these features.

**ICNR initialization for PixelShuffle:** Checkerboard artifacts are a known failure mode of PixelShuffle upsampling when initialized randomly. ICNR (Integer Convolution Nearest-Neighbour Resize) initializes the PixelShuffle convolution weights so that the initial output is equivalent to nearest-neighbour upsampling, entirely eliminating the artifact at initialization.

**U-Net discriminator over PatchGAN:** A standard PatchGAN produces a coarse spatial grid of real/fake predictions, providing limited gradient information. The U-Net discriminator produces a dense 128x128 per-pixel map via skip connections, giving the generator richer and more spatially precise feedback.

**EMA generator at inference:** Exponential moving averaging of generator weights reduces the noise introduced by mini-batch gradient updates. With decay=0.9995 over 400 total epochs, the EMA model effectively averages over approximately 2,000 gradient steps, producing consistently lower MAE than the raw checkpoint.

**Conservative Phase 2 adversarial weight (0.0002):** GAN fine-tuning was designed to improve perceptual texture quality without degrading the pixel accuracy achieved in Phase 1. Setting the adversarial weight two orders of magnitude below the pixel weight ensured that the discriminator could improve sharpness while Phase 2 MAE remained within 0.02 of the Phase 1 best.

**Best-checkpoint saving:** Both training phases saved the model checkpoint only when the current-epoch MAE was strictly lower than all previous epochs, ensuring that late-epoch regression (which can occur near learning rate restarts) does not affect the final model.

---

## Environment and Dependencies

The notebook was developed and executed on Kaggle using the standard `kaggle/python` Docker image.

| Dependency     | Purpose                                      |
|----------------|----------------------------------------------|
| Python 3.x     | Core language                                |
| PyTorch        | Model definition, training, and inference    |
| torchvision    | VGG-19 model structure                       |
| NumPy          | Array operations and metric computation      |
| Pillow (PIL)   | Image loading and resizing                   |
| tqdm           | Progress bar display during training         |
| matplotlib     | Training curve plots and EDA visualizations  |
| pandas         | Submission CSV construction                  |

**Hardware:** CUDA GPU (Kaggle-provided). Mixed precision training (`torch.amp.autocast` and `GradScaler`) was used throughout to maximize throughput and reduce memory consumption.

**Reproducibility settings:**

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False   # kept False for throughput
torch.backends.cudnn.benchmark = True
```

Note that `cudnn.deterministic = False` was intentionally kept disabled to preserve training throughput; results may vary slightly across runs at the GPU operation level.

---

## Repository Structure

```
.
|-- 24f1002299-notebook-26t1-nppe2.ipynb   # Full Kaggle notebook with all training code and outputs
|-- Plant_Leaves_Super_Resolution_Report.pdf # Compiled technical report (PDF)
|-- 24f1002299_report.tex                  # LaTeX source for the technical report
|-- 24f1002299_code.md                     # Extracted implementation code (all notebook cells)
|-- README.md                              # This file
|-- .gitignore                             # Git ignore rules
```

The notebook is self-contained and can be re-executed on Kaggle by attaching the `plant-leaves-super-resolution-challenge` competition dataset. All intermediate checkpoints (`best_generator_phase1.pth`, `best_ema_phase1.pth`, etc.) are saved to the Kaggle working directory during training and are not tracked in this repository.

---

## References

1. Kaggle Plant Leaves Super-Resolution Challenge dataset (AgriVision).
2. X. Wang et al., *ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks*, ECCV Workshops, 2018.
3. O. Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI, 2015.
4. K. Simonyan and A. Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition* (VGG-19), ICLR, 2015.
5. I. Loshchilov and F. Hutter, *Decoupled Weight Decay Regularization* (AdamW), ICLR, 2019.
6. T. R. Dozat, *Incorporating Nesterov Momentum into Adam*, ICLR Workshop, 2016.
7. PyTorch documentation: https://pytorch.org/docs/stable/
