# Project Guide

## Overview

This is a PyTorch neural style transfer project. It includes the classic Gatys-style optimization pipeline and a fast feed-forward style transfer training/inference pipeline.

## Setup

- Install dependencies with `pip install -r requirements.txt`.
- Use `conda activate DL` when following the commands documented in `README.md`.

## Source Layout

- `run_style.py` runs optimization-based neural style transfer.
- `train_fast_style.py` trains a `TransformerNet` fast style transfer model.
- `stylize_fast.py` applies a trained fast style checkpoint.
- `libs/` contains dataset helpers, image preprocessing, VGG19 feature extraction, loss functions, and the transformation network.
- `data/content/` and `data/style/` store example input images.
- `data/train/content/val2017/` stores COCO validation images for training experiments.
- `outputs/` and `checkpoints/` contain generated artifacts and trained weights.

## Validation

- Smoke test classic NST with:
  `python run_style.py --content data/content/city_street.jpg --style data/style/starry_night.jpg --out outputs/result.png --steps 5`
- Smoke test fast model training with:
  `python train_fast_style.py --content-dir data/train/content/val2017 --style data/style/starry_night.jpg --epochs 1 --batch-size 2 --limit-images 16 --sample-interval 2 --log-interval 1`

## Notes

- VGG19 weights are downloaded through `torchvision` by default; local weights can be placed under `models/` and passed with `--vgg-weights`.
- Generated images and checkpoints can be large; avoid committing bulky dataset, output, or checkpoint artifacts unless explicitly needed.
