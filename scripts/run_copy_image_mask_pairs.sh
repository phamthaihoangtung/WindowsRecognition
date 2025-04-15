#!/bin/bash

# Define variables for input and output directories
IMAGES_DIR="data/processed/v1/images/train"
MASKS_DIR="data/processed/v1/annotations/train"
OUTPUT_IMAGES_DIR="data/processed/sanity_check/images/train"
OUTPUT_MASKS_DIR="data/processed/sanity_check/annotations/train"

# Run the Python script using uv
uv run python utils/copy_image_mask_pairs.py "$IMAGES_DIR" "$MASKS_DIR" "$OUTPUT_IMAGES_DIR" "$OUTPUT_MASKS_DIR" --sample_count 200
