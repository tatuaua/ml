# Simple Feed-Forward Neural Network for Text Generation

A minimal FNN that predicts the next token given a context window of previous tokens.

## How It Works

1. Take the last N tokens as context
2. Convert each token to a vector (embedding lookup)
3. Flatten into one long vector
4. Pass through hidden layer with activation
5. Project to vocabulary size
6. Convert to probabilities
7. Pick the most likely next token

## Architecture

```
Input: [token_1, token_2, ..., token_N]
         ↓
    Embedding Lookup
         ↓
    [N × embed_dim] matrix
         ↓
    Flatten to 1D vector
         ↓
    Linear Layer (weights + bias)
         ↓
    ReLU Activation
         ↓
    Linear Layer (weights + bias)
         ↓
    Softmax → probabilities
         ↓
    Argmax → next token
```

## Required Functions

| Function | Purpose |
|----------|---------|
| **Lookup** | Retrieve embedding vectors by token ID |
| **Flatten** | Reshape 2D tensor to 1D |
| **MatMul** | Matrix multiplication for linear layers |
| **Add** | Add bias vectors |
| **ReLU** | Activation function: max(0, x) |
| **Softmax** | Convert logits to probabilities |
| **Argmax** | Select highest probability token |

## Key Concepts

**Embedding**: A lookup table mapping token IDs to dense vectors. Each row is a learnable representation of one token.

**Linear Layer**: `output = input × weights + bias`. The core building block of neural networks.

**ReLU**: Introduces non-linearity. Without it, stacked linear layers collapse to a single linear transformation.

**Softmax**: Converts raw scores to a probability distribution that sums to 1. Uses exp(x)/sum(exp(x)) with numerical stability tricks.

**Argmax**: Greedy decoding—simply pick the token with highest probability.

## Limitations

- Fixed context window (no long-range memory)
- No attention mechanism
- Quality depends heavily on training data and hyperparameters
- This implementation is inference-only (no training)
