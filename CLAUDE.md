# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install uv
uv sync

# Install SAM2 submodule
git submodule update --init --recursive
bash scripts/install_sam.sh
```

Add a `.env` file with your Weights & Biases API key:
```
WANDB_API_KEY=your_wandb_api_key_here
```

## Commands

```bash
# Train
uv run python src/train.py

# Run inference server (Flask, port 5001)
uv run python src/server.py

# Batch inference with SAM
uv run python src/inference_sam.py
```

Scripts mirror the above:
- `scripts/train.sh` — training
- `scripts/infer_sam.sh` — batch SAM inference
- `scripts/infer.sh` — standard inference

## Architecture

The system is a two-stage window segmentation pipeline:

**Stage 1 — Coarse segmentation** (`src/model.py`, `src/train.py`):
- PyTorch Lightning `SegmentationModel` wrapping `segmentation_models_pytorch` (Unet or FPN with configurable EfficientNet backbone).
- Trained with a combined loss (see `src/loss.py`), evaluated by IoU.
- Config is loaded from `config/config.yaml`; model checkpoints saved to `models/<experiment_name>/`.

**Stage 2 — SAM2 refinement** (`src/inference_sam.py`, `src/inference/`):
- `WindowsRecognitor` runs the segmentation model, then refines each detected window region with SAM2.
- Two refinement modes (set via `config.inference.refined_segmentation_mode`):
  - `"contour"` (default): finds contours in the coarse mask → crops each window region → iteratively prompts SAM2 (`SAM2ImagePredictor`) with positive/negative points → averages multiple sampling runs.
  - `"tiling"`: uses Ultralytics SAM with a tiling approach.
- Post-processing (`src/inference/post_processor.py`): removes small regions, applies polygon simplification, convex hull approximation.

**Inference server** (`src/server.py`):
- Flask app on port 5001 with a single `POST /upload-image` endpoint.
- Expects `{"path": "<filename>"}` where the file lives in `UPLOADED_IMAGE_DIR`.
- Saves binary masks to `PROCESSED_IMAGE_DIR` and serves them at `GET /processed_images/<filename>`.
- Paths (`D:/server01/...`) are hardcoded for the production Windows server — change for local use.

## Configuration

All configs live in `config/`. Key fields:
- `config.yaml` — base training config
- `config_prod.yaml` — production inference (used by `server.py`)
- `config_sam.yaml` / `config_large.yaml` — experiment variants

`inference.refined_segmentation_mode` controls which SAM refinement path is used. `inference.sam_checkpoint` sets the HuggingFace model ID (default `facebook/sam2.1-hiera-large`).

## Data layout

```
data/processed/<version>/
  images/{train,val,test}/
  annotations/{train,val,test}/
```

The config `paths` section points to these directories. Data is not tracked in git.
