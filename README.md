# WindowsRecognition

## Installation

Before setting up the environment, ensure you have `uv` installed:

```bash
pip install uv
```

Then, follow the steps to configure and activate the environment:

```bash
uv sync
```

## Configuration

Edit the `config/config.yaml` file to set paths, hyperparameters, and model configurations.

## Training

1. Add your Weights & Biases API key to the `.env` file:

```plaintext
WANDB_API_KEY=your_wandb_api_key_here
```

2. Run the training script:

```bash
uv run python src/train.py
```