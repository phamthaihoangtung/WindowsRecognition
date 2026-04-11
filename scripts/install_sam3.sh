git submodule update --init --recursive

export $(grep -v '^#' .env | xargs)
uv run huggingface-cli login --token "$HF_TOKEN"

# cc_torch requires nvcc and CUDA headers from the nvidia pip wheels
VENV_SITE=$(uv run --no-sync python -c "import site; print(site.getsitepackages()[0])")
CUDA_HOME=/usr/local/cuda-12.6 \
CPLUS_INCLUDE_PATH=$(find "$VENV_SITE/nvidia" -name "include" -type d | tr '\n' ':') \
uv sync
