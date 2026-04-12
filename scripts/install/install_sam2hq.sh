#!/bin/bash
# Install SAM2-HQ checkpoint for the SAMRefiner sam2_hq backend.
# The SAM2-HQ model code lives in samrefiner/sam-hq/sam-hq2/ (submodule).
# This script only downloads the checkpoint weights.
set -euo pipefail

CKPT_DIR="models/samrefiner"
CKPT_FILE="${CKPT_DIR}/sam2.1_hq_hiera_l.pt"
HF_URL="https://huggingface.co/lkeab/hq-sam/resolve/main/sam2.1_hq_hiera_large.pt"

mkdir -p "${CKPT_DIR}"

if [ -f "${CKPT_FILE}" ]; then
    echo "Checkpoint already exists: ${CKPT_FILE}"
    exit 0
fi

echo "Downloading SAM2.1-HQ Large checkpoint..."
wget -q --show-progress "${HF_URL}" -O "${CKPT_FILE}"
echo "Saved to ${CKPT_FILE}"
