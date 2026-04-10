git submodule update --init --recursive

export $(grep -v '^#' .env | xargs)
uv run huggingface-cli login --token "$HF_TOKEN"

uv sync
