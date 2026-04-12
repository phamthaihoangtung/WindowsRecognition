# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install uv
uv sync

# Install SAM2 submodule
git submodule update --init --recursive
bash scripts/install/install_sam.sh
```

Add a `.env` file with your Weights & Biases API key:
```
WANDB_API_KEY=your_wandb_api_key_here
```

Optional external refiners (install only what you need):
```bash
bash scripts/install/install_sam3.sh       # SAM3 coarse segmentation
bash scripts/install/install_cascadepsp.sh # CascadePSP stage-2 refiner
bash scripts/install/install_crm.sh        # CRM stage-2 refiner
bash scripts/install/install_samrefiner.sh # SAMRefiner (SAM-HQ) stage-2 refiner
bash scripts/install/install_segrefiner.sh # SegRefiner (diffusion-based) stage-2 refiner
```

## Commands

```bash
# Train
uv run python src/train.py

# Run inference server (Flask, port 5001)
uv run python src/server.py

# Batch inference — pick the right script for the mode you want:
uv run python src/inference_sam.py   # general entry point
```

Scripts:
- `scripts/train.sh` — training
- `scripts/infer/infer.sh` — standard inference
- `scripts/infer/infer_sam.sh` — SAM2 contour/tiling refinement
- `scripts/infer/infer_sam3.sh` — SAM3 coarse + SAM2 refinement
- `scripts/infer/infer_cascadepsp.sh` — SAM3 coarse + CascadePSP refinement
- `scripts/infer/infer_crm.sh` — SAM3 coarse + CRM refinement
- `scripts/infer/infer_samrefiner.sh` — SAM3 coarse + SAMRefiner (SAM-HQ) refinement
- `scripts/infer/infer_segrefiner.sh` — SAM3 coarse + SegRefiner (diffusion) refinement
- `scripts/infer/infer_none.sh` — SAM3 coarse + direct post-processing (no Stage 2)

Data utilities: `scripts/data/` — crawl_interiors, move_matching_masks, run_copy_image_mask_pairs

## Architecture

The system is a two-stage window segmentation pipeline:

**Stage 1 — Coarse segmentation** (set via `config.inference.coarse_segmentation_mode`):
- `"efficientnet"` (default): PyTorch Lightning `SegmentationModel` wrapping `segmentation_models_pytorch` (Unet or FPN). Config in `config/config.yaml`; checkpoints saved to `models/<experiment_name>/`. See `src/model.py`, `src/train.py`.
- `"sam3"`: Open-vocabulary detection using SAM3 with a text prompt (`sam3_text_prompt`). Produces a probability map from scored instance detections. `sam3_fp16` (bool, default false) runs activations under `torch.autocast` fp16 (CUDA only). Note: model weight quantization (`.half()`) is incompatible with SAM3 — autocast preserves some ops in fp32 (LayerNorm, softmax) which then mismatches fp16 weights.

**Stage 2 — Refinement** (`src/inference_sam.py`, `src/inference/`):
- `WindowsRecognitor` runs Stage 1, then passes the coarse mask to the configured Stage 2 refiner.
- Mode is set via `config.inference.refined_segmentation_mode`:
  - `"contour"` (default): finds contours in the coarse mask → crops each window region → iteratively prompts SAM2 (`SAM2ImagePredictor`) with positive/negative points → averages multiple sampling runs.
  - `"tiling"`: uses Ultralytics SAM with a tiling approach.
  - `"cascadepsp"`: global + local boundary refinement via CascadePSP. Keys: `cascadepsp_threshold`, `cascadepsp_L`, `cascadepsp_fast`, `cascadepsp_fp16` (CUDA-only autocast). See `src/inference/cascade_processor.py`.
  - `"crm"`: multi-scale LIIF-based refinement via CRMNet. Keys: `crm_threshold`, `crm_scales`, `crm_checkpoint`. See `src/inference/crm_processor.py`.
  - `"samrefiner"`: iterative SAM-HQ refinement using geodesic point sampling and mask prompts. Keys: `samrefiner_checkpoint`, `samrefiner_model_type`, `samrefiner_iters`, `samrefiner_gamma`, `samrefiner_strength`, `samrefiner_margin`, `samrefiner_threshold`, `samrefiner_use_point/box/mask/add_neg`. See `src/inference/samrefiner_processor.py`.
  - `"segrefiner"`: discrete diffusion-based refinement via SegRefiner (DenoiseUNet, semantic mode). Keys: `segrefiner_config`, `segrefiner_checkpoint`, `segrefiner_threshold`. See `src/inference/segrefiner_processor.py`.
  - `"none"`: skip Stage 2 entirely; coarse mask is passed directly to post-processing.
- Post-processing (`src/inference/post_processor.py`): removes small regions, applies polygon simplification, convex hull approximation. `cascadepsp` uses Otsu binarization; `crm`, `samrefiner`, and `segrefiner` bypass with a hard `>= 0.5` threshold. `dilation_kernel_size` (general config key) dilates per-contour before Douglas-Peucker only, compensating for its inward-cutting bias.

**Inference server** (`src/server.py`):
- Flask app on port 5001 with a single `POST /upload-image` endpoint.
- Expects `{"path": "<filename>"}` where the file lives in `UPLOADED_IMAGE_DIR`.
- Saves binary masks to `PROCESSED_IMAGE_DIR` and serves them at `GET /processed_images/<filename>`.
- Paths (`D:/server01/...`) are hardcoded for the production Windows server — change for local use.

## Configuration

All configs live in `config/`. Key fields:

| File | Purpose |
|---|---|
| `config.yaml` | Base training config |
| `config_prod.yaml` | Production inference (used by `server.py`) |
| `config_sam.yaml` / `config_large.yaml` | SAM2 experiment variants |
| `config_sam3.yaml` | SAM3 coarse + SAM2 contour refinement |
| `config_cascadepsp.yaml` | SAM3 coarse + CascadePSP refinement |
| `config_crm.yaml` | SAM3 coarse + CRM refinement |
| `config_samrefiner.yaml` | SAM3 coarse + SAMRefiner (SAM-HQ) refinement |
| `config_segrefiner.yaml` | SAM3 coarse + SegRefiner (diffusion) refinement |
| `config_none.yaml` | SAM3 coarse + direct post-processing (no Stage 2) |

`inference.coarse_segmentation_mode` selects Stage 1 (`"efficientnet"` or `"sam3"`).
`inference.refined_segmentation_mode` selects Stage 2 (`"contour"`, `"tiling"`, `"cascadepsp"`, `"crm"`, `"samrefiner"`, `"segrefiner"`, or `"none"` to skip Stage 2).
`inference.stage2_input_size` resizes the smallest dimension before Stage 2 (`null` = keep Stage 1 output size).

## Data layout

```
data/processed/<version>/
  images/{train,val,test}/
  annotations/{train,val,test}/
```

The config `paths` section points to these directories. Data is not tracked in git.
