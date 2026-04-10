# WindowsRecognition

## Installation

Before setting up the environment, ensure you have `uv` installed:

```bash
pip install uv
```

Then, follow the steps to configure and activate the environment:

```bash
bash scripts/install_sam.sh
```

This initializes the SAM2 submodule and runs `uv sync` to install all dependencies including SAM2.

### SAM3 (experimental)
Add `HF_TOKEN=your_token_here` to your `.env` file — the model is gated on HuggingFace.

```bash
bash scripts/install_sam3.sh
```

This initializes the SAM3 submodule, authenticates with HuggingFace, and runs `uv sync` to install all dependencies including SAM3.

## Configuration

Edit the config under `config` file to set paths, hyperparameters, and model configurations.

## Training

1. Add your Weights & Biases API key to the `.env` file:

```plaintext
WANDB_API_KEY=your_wandb_api_key_here
```

2. Run the training script:

```bash
uv run python src/train.py
```

## Architecture
<img width="1733" height="1057" alt="Architecture" src="https://github.com/user-attachments/assets/f236b8f6-6bc5-4499-9688-5797451e74a9" />
### Inference

Run the inference server
```bash
uv run python src/server.py 
```
