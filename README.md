# Lab of Future — Multi-Task Brain Tumor Segmentation & Classification

A from-scratch PyTorch implementation of a multi-task U-Net that simultaneously performs **pixel-level tumor segmentation** and **4-class tumor classification** on brain MRI scans, using a single shared encoder-decoder backbone.

## Overview

This project trains one U-Net-based model to solve two tasks at once from a single MRI slice:

1. **Segmentation** — predict a binary mask localizing the tumor region, pixel by pixel.
2. **Classification** — predict which of 4 categories the scan belongs to: `glioma`, `meningioma`, `pituitary`, or `notumor`.

Rather than training two separate single-task models, this implementation shares a single encoder/bottleneck across both heads — the bottleneck feature map feeds both the decoder (for segmentation) and a lightweight classification head (via global average pooling + MLP), so the model learns one shared representation useful for both tasks.

## Architecture

- **Encoder**: 4 downsampling stages, each a double-conv block (`Conv2d → ReLU → Conv2d → ReLU`) followed by 2×2 max pooling. Channel widths: 64 → 128 → 256 → 512.
- **Bottleneck**: double-conv block at the lowest spatial resolution, expanding to 1024 channels.
- **Segmentation decoder**: 4 upsampling stages using transposed convolutions, with skip connections concatenating corresponding encoder feature maps at each resolution (standard U-Net skip-connection design), followed by a final 1×1 conv producing a single-channel raw logit mask.
- **Classification head**: global average pooling over the bottleneck feature map → flatten → 2-layer MLP (`Linear → ReLU → Linear`) producing 4-class logits.
- **Input**: single-channel (grayscale) 256×256 MRI slices.
- **Output**: `(mask_logits, class_logits)` — a tuple of raw, unnormalized outputs. Sigmoid/argmax are applied downstream during inference and evaluation, not inside the model.

## Dataset

Sourced from the [Brain Tumor Dataset: Segmentation & Classification](https://www.kaggle.com/datasets/indk214/brain-tumor-dataset-segmentation-and-classification) on Kaggle, which ships as two independent folder structures:

- **`Segmentation/`** — tumor-positive scans only, organized by class (`Glioma`, `Meningioma`, `Pituitary tumor`), each image paired with a binary mask (`<name>.png` / `<name>_mask.png`).
- **`classification/{Training,Testing}/notumor/`** — tumor-negative scans, image-only (no masks, since there's no tumor to localize).

Since these two folder structures don't share overlapping images or a consistent 4-class labeling scheme, this project builds one unified dataset by:

- Pulling all 3 tumor classes from `Segmentation/` with their real masks.
- Pulling all `notumor` images from **both** the `Training` and `Testing` classification folders, assigning each an all-zero mask (correct ground truth for "no tumor present").

**Class distribution** (unified dataset):

| Class | Samples |
|---|---|
| Glioma | 554 |
| Meningioma | 708 |
| Pituitary | 930 |
| No Tumor | 2,000 |
| **Total** | **4,192** |

Note the class imbalance — `notumor` outnumbers the smallest tumor class (`glioma`) by roughly 3.6×. This is handled by stratified splitting and macro-averaged classification metrics (see below).

## Data Pipeline

- **Stratified 70/15/15 train/val/test split**, stratified by class label, ensuring all splits preserve the same class ratios as the full dataset — important given the imbalance above. Split is index-based with a fixed random seed for full reproducibility.
- **Augmentation** (training only, via `albumentations`, applied jointly to image + mask so spatial transforms stay aligned):
  - Resize to 256×256
  - Horizontal flip (p=0.5)
  - Rotation (±15°, p=0.5)
  - Random brightness/contrast (p=0.3)
  - Normalization (mean=0.5, std=0.5 for single-channel input)
- **Validation/test** use resize + normalization only — no augmentation, to reflect real, unmodified model performance.

## Training

- **Loss**: weighted sum of `BCEWithLogitsLoss` (segmentation) + `CrossEntropyLoss` (classification), unweighted (1:1) since both losses trained at comparable scale without one task starving the other.
- **Optimizer**: Adam, learning rate `1e-4`.
- **Epochs**: 20.
- **Checkpointing**: best model selected by highest validation Dice score (segmentation is the primary task), saved via `state_dict()`.
- **Metrics tracked per epoch**: train loss, validation loss, validation Dice, validation IoU, validation classification accuracy, validation F1 (macro-averaged across classes to account for class imbalance).

## Results

Final test-set evaluation (best checkpoint, selected by validation Dice):

| Metric | Score |
|---|---|
| Test Dice Score | **0.6787** |
| Test IoU (Jaccard Index) | **0.5137** |
| Test Classification Accuracy | **0.9253** |
| Test Classification F1 (macro) | **0.8809** |

**Honest read on these numbers**: classification performance is strong and consistent between validation and test (no overfitting gap), suggesting the shared backbone learns a solid representation for distinguishing tumor types. Segmentation Dice (~0.68) is a reasonable result for a from-scratch U-Net trained for 20 epochs without extensive hyperparameter tuning, but had not fully converged — validation Dice was still improving/fluctuating at the final epoch. Likely next steps for improvement: more training epochs, `pos_weight` adjustment in the segmentation loss to better handle pixel-level foreground/background imbalance, and exploring uncertainty-based loss weighting (e.g., Kendall et al.) if one task is found to dominate gradient updates.

## Project Structure

```
.
├── UNet_PyTorch_Implementation.ipynb   # Model definition, dataset/dataloader, training loop
├── DATASET/
│   ├── Segmentation/
│   │   ├── Glioma/
│   │   ├── Meningioma/
│   │   └── Pituitary tumor/
│   └── classification/
│       ├── Training/
│       └── Testing/
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` pins CUDA 12.6 PyTorch wheels (`torch`, `torchvision`, `torchaudio`) via the PyTorch extra index, plus the core data/ML stack (`albumentations`, `opencv-python`, `scikit-learn`, `numpy`, `pandas`) and Jupyter (`jupyter`, `ipykernel`, `ipywidgets`) for running the notebook. If you're on CPU-only or a different CUDA version, replace the `--extra-index-url` line and the three pinned `torch*` versions with the build matching your setup — see [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

Download the dataset from Kaggle and place it under `DATASET/` matching the structure above.

## Usage

Open `UNet_PyTorch_Implementation.ipynb` and run cells top to bottom:

1. Dataset/DataLoader construction with stratified split
2. Model instantiation
3. Training loop (20 epochs, checkpoints best model to `best_model.pth`)
4. Test-set evaluation using the saved best checkpoint

## Tech Stack

**Core**: `PyTorch` (CUDA 12.6) · `torchvision` · `albumentations` · `OpenCV` · `scikit-learn` · `NumPy` · `pandas`

**Tooling**: `Jupyter` / `ipykernel` for development, `torchinfo` for model summaries, `matplotlib` for visualizing samples/masks

## Future Improvements

- Extend training beyond 20 epochs with learning rate scheduling to reach full convergence
- Address pixel-level class imbalance in segmentation via `pos_weight` or a Dice/Focal loss variant
- Explore learned (uncertainty-based) task loss weighting if fixed 1:1 weighting shows one task lagging
- Add train-set metric tracking alongside validation to monitor for overfitting over longer training runs
