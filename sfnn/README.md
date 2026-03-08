# Simple Feed-Forward Neural Network (Go)

Character-level feed-forward neural network for next-token prediction and text generation.

## Current Status

Implemented and working in the current codebase:

- Character vocabulary building and encode/decode
- Sliding-window dataset creation
- Embedding lookup + flatten
- Two-layer MLP (`Linear -> ReLU -> Linear`)
- Softmax + cross-entropy loss
- Full backward pass (manual gradients)
- SGD updates with gradient clipping
- Mini-batch training loop with per-batch logging
- Greedy generation with `ArgMax`
- Parallelized tensor ops / forward-backward hotspots using goroutines

## Model Architecture

```
Input context (N chars)
  -> Embedding lookup
  -> Flatten
  -> Linear (W1, B1)
  -> ReLU
  -> Linear (W2, B2)
  -> Softmax probabilities
  -> ArgMax next token
```

## Default Hyperparameters

- `contextSize = 8`
- `embedDim = 10`
- `hiddenDim = 128`
- `batchSize = 32`
- `learningRate = 0.1`
- `epochs = 10`
- `clipThresh = 5.0`
- `genLength = 100`

Defined in `main.go`.

## Project Files

- `main.go` - data loading, model definition, forward/backward, training loop, generation
- `tensor.go` - tensor type and ops (`MatMul`, `Lookup`, `Flatten`, `ReLU`, `SoftMax`, `ArgMax`, etc.)
- `tensor_test.go` - unit tests for core tensor operations
- `data/input.txt` - training corpus
- `run.ps1` - runs tests and then starts training/generation

## Run

### PowerShell

```powershell
./run.ps1
```

### Manual

```powershell
go test ./...
go run .
```

## Notes / Limitations

- Fixed context window (no long-range memory)
- Character-level model (not tokenized words/subwords)
- Greedy decoding only (no sampling/top-k/top-p)
- No checkpoint save/load yet
