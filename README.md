# Mamba Autoencoder Transformer

A PyTorch implementation of a hybrid Transformer model in an autoencoder configuration, featuring a bottleneck architecture and a mix of standard attention and Mamba (Selective SSM) blocks.

## Overview

This project explores a hybrid neural architecture that combines the strengths of standard Transformer attention with the efficiency of Mamba blocks. The model is structured as an autoencoder, where the embedding dimension varies across layers to create a bottleneck effect, potentially useful for sequence compression or feature extraction tasks.

## Key Features

- **Hybrid Layer Architecture**: Each layer can be independently configured as either a standard Multi-Head Attention block or a simplified Mamba (Selective SSM) block.
- **Autoencoder Bottleneck**: Customizable embedding dimensions (`n_embd`) per layer allow for "hourglass" architectures where the representation is compressed and then expanded.
- **Sequential Data Loading**: A specialized `DataLoader` designed for training on long-form text (e.g., books), maintaining state across batches—essential for the recurrent nature of Mamba blocks.
- **Configuration Driven**: All hyperparameters and hardware settings are managed via `current_configuration.json` for easy experimentation.

## Architecture

### Model Components (`model.py`)
- **`GPT`**: The top-layer container managing token and position embeddings, a sequence of blocks, and the final language modeling head.
- **`Block`**: A wrapper that routes input through either a `MultiHeadAttention` or `MambaBlock`, followed by a `FeedForward` layer.
- **`MambaBlock`**: A simplified implementation of the Selective State Space Model, including recursive scanning and 1D convolutions.
- **`MultiHeadAttention`**: Standard causal self-attention implementation.

### Bottleneck Configuration
The model allows passing a list for `n_embd` and `n_head`. For example:
```json
"n_embd": [256, 256, 192, 128, 128, 192, 256, 256],
"n_layer": 8
```
This creates a bottleneck where middle layers have reduced capacity.

## Installation

This project uses `uv` for dependency management.

1. Clone the repository.
2. Install dependencies:
   ```bash
   uv sync
   ```

## Usage

### 1. Dataset Preparation
Place your `.txt` files in the `dataset/` directory. The `DataLoader` will automatically tokenize and cache the data.

### 2. Configuration
Review and modify `current_configuration.json`. If it doesn't exist, running `train.py` once will generate a default one:
```json
{
    "config": {
        "batch_size": 64,
        "block_size": 128,
        "n_embd": [256, 256, 192, 128, 128, 192, 256, 256],
        "attention_type": ["mamba", "standard", "standard", "mamba", "standard", "standard", "standard", "standard"]
    },
    "device": "cuda",
    "compile": true
}
```

### 3. Training
Start the training process:
```bash
uv run train.py
```
The script will periodically evaluate loss, save checkpoints (`checkpoint_step_*.pt`), and generate sample text to monitor progress.

## Dependencies

- Python >= 3.13
- PyTorch >= 2.9.1
- Tiktoken (GPT-2 encoding)
- NumPy

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
