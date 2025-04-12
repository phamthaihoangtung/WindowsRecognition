#!/bin/bash

# Define variables for input and output directories
IMAGES_DIR="data/staging/split/test"
MASKS_DIR="data/raw/mask"
OUTPUT_IMAGES_DIR="data/processed/v1/images/test"
OUTPUT_MASKS_DIR="data/processed/v1/annotations/test"

# Run the Python script using uv
uv run python src/utils/copy_image_mask_pairs.py "$IMAGES_DIR" "$MASKS_DIR" "$OUTPUT_IMAGES_DIR" "$OUTPUT_MASKS_DIR"
