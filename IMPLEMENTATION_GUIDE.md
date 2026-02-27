# Character-Level LLM Big Bang (FNN) — Implementation Guide (No Code)

This document is a practical, beginner-friendly build plan for your Go project.

## End Goal

By the end, you can:
- Train a simple character-level Feed-Forward Neural Network on `data/input.txt`.
- Hardcode a prompt in `main.go` (for example, `ROMEO:`).
- Run the app and print generated text (100+ characters) to the terminal.

## What You Are Building (Mental Model)

You are building a **next-character predictor**:
- Input: the last `N` characters (context window).
- Output: scores for every character in your vocabulary.
- Decision: choose the character with the highest score (greedy decoding).
- Repeat: append the new character and slide the window forward.

This is exactly the core of “next-token prediction,” just at character level and with an FNN instead of a Transformer.

## Fixed Hyperparameters (Use These)

- Context Size (`N`): 8
- Embedding Dimension: 10
- Hidden Dimension: 128
- Batch Size: 32
- Learning Rate: 0.1
- Epochs: start with 10, then try up to 20
- Gradient Clip Threshold: start with 1.0 or 5.0 (either is fine if kept consistent)

## Project Layout You Should End With

Keep your existing files and add structure logically:

- `data/input.txt`: training corpus (Shakespeare text)
- `tensor.go`: tensor operations (already partially implemented)
- `tensor_test.go`: unit tests for math correctness
- `main.go`: orchestrates training + generation and prints final text

Recommended conceptual modules (can be separate files later, but not required immediately):
- Data/vocabulary pipeline
- Model parameters and forward pass
- Loss and backward pass
- Optimizer (SGD)
- Text generation

---

## Sprint 1: Data & Vocabulary

### Task 1: Build the Character Map (Vocabulary)

Objective: convert each unique character into a stable integer ID.

Steps:
1. Read all text from `data/input.txt` as raw characters.
2. Collect unique characters from the full corpus.
3. Sort them in a deterministic order (important for reproducibility).
4. Create two maps:
   - character → integer ID
   - integer ID → character
5. Store `vocabSize` as the total unique character count.

Why this matters:
- Every later tensor shape depends on `vocabSize`.
- Deterministic ordering ensures consistent runs and easier debugging.

Definition check:
- **Vocabulary** = all unique characters the model can predict.

Acceptance criteria:
- You can print the vocabulary size.
- You can encode and decode any character from the corpus without mismatch.

---

### Task 2: Build Sliding Window Dataset

Objective: create `(context, target)` training pairs.

Given `N=8`, each training sample is:
- Input context: 8 character IDs.
- Target label: the next character ID.

Steps:
1. Use the fully encoded corpus (from Task 3 below).
2. For each position `i` from `0` to `len(data)-N-1`:
   - context = positions `[i : i+N]`
   - target = position `[i+N]`
3. Save all pairs into arrays/slices suitable for batching.
4. Optionally shuffle pair order each epoch for more stable learning.

Definition check:
- **Context Window (`N`)** = number of previous characters used to predict the next one.

Acceptance criteria:
- First few samples look logically correct when decoded back to text.
- Number of samples equals `len(encodedText) - N`.

---

### Task 3: Numerical Encoding of Full Text

Objective: convert entire corpus to integer IDs once and reuse it.

Steps:
1. Iterate through each character in input text.
2. Replace each character by its vocabulary ID.
3. Keep the encoded sequence in memory as one long integer list.
4. Validate by decoding first 100 IDs back to text and comparing with source.

Acceptance criteria:
- Encode → decode roundtrip reproduces original text exactly.

---

## Sprint 2: Model Architecture

### Model Shape Overview

For one training sample:
- Input IDs shape: `(N)`
- After embedding lookup: `(N, embedDim)`
- After flatten: `(N * embedDim)`
- Hidden pre-activation: `(hiddenDim)`
- Hidden post-ReLU: `(hiddenDim)`
- Logits: `(vocabSize)`

For a batch of size `B`:
- Input IDs: `(B, N)`
- Embeddings: `(B, N, embedDim)`
- Flattened: `(B, N*embedDim)`
- Hidden: `(B, hiddenDim)`
- Logits: `(B, vocabSize)`

---

### Task 4: Embedding Layer

Objective: map token IDs to dense vectors.

Parameter to create:
- Embedding matrix shape: `(vocabSize, embedDim)`

Initialization guidance:
- Small random values centered near zero.
- Keep scale modest to avoid unstable logits early.

Forward behavior:
- For each token ID in the context, pick corresponding embedding row.
- Concatenate (flatten) all `N` vectors into one input vector.

Definition check:
- **Embedding Dim** = number of numeric features per character.

Acceptance criteria:
- Input of `N=8` IDs yields flattened vector of length `8 * 10 = 80`.

---

### Task 5: Hidden Layer (Linear + ReLU)

Objective: add non-linear representation power.

Parameters:
- `W1` shape: `(N*embedDim, hiddenDim)` = `(80, 128)`
- `b1` shape: `(hiddenDim)` = `(128)`

Forward behavior:
1. Linear transform: `h_pre = x·W1 + b1`
2. Activation: `h = ReLU(h_pre)`

Definition check:
- **Hidden Dim** = model capacity in the middle layer.

Acceptance criteria:
- Hidden output shape is exactly `(128)` for one sample.
- ReLU zeroes negative values.

---

### Task 6: Output Layer (Logits)

Objective: produce one score per vocabulary character.

Parameters:
- `W2` shape: `(hiddenDim, vocabSize)`
- `b2` shape: `(vocabSize)`

Forward behavior:
- `logits = h·W2 + b2`

Interpretation:
- Logits are raw scores, not probabilities.
- Apply softmax only for probabilities or loss calculation.

Acceptance criteria:
- Logit vector length equals `vocabSize`.
- Highest logit index can be decoded to a character.

---

## Sprint 3: Training Loop

### Task 7: Cross-Entropy Loss

Objective: measure how wrong predictions are.

For each sample:
1. Compute logits for vocab.
2. Convert to probabilities with numerically stable softmax.
3. Select probability of true target ID.
4. Loss = negative log of that probability.

For batch:
- Mean loss over all samples in the batch.

Why cross-entropy:
- It directly rewards assigning high probability to correct next character.

Acceptance criteria:
- Loss is positive and finite.
- Early in training loss is high; it should generally trend downward.

---

### Task 8: Backpropagation + SGD

Objective: update parameters to reduce loss.

Parameters to train:
- Embedding matrix
- `W1`, `b1`
- `W2`, `b2`

High-level backward flow:
1. Compute gradient at logits using cross-entropy derivative.
2. Backprop through output linear layer to hidden layer.
3. Backprop through ReLU.
4. Backprop through first linear layer.
5. Accumulate gradients for embedding rows used by token IDs.

SGD update rule:
- For each parameter tensor: subtract `learningRate * gradient`.

Definition check:
- **SGD** = simple optimizer that moves parameters opposite gradient.
- **Learning Rate** = size of each parameter step.

Acceptance criteria:
- Parameters change every training step.
- Average epoch loss decreases over time.

---

### Task 9: Gradient Clipping

Objective: keep updates stable when gradients explode.

Steps:
1. Before optimizer step, examine gradient values (or total norm).
2. If magnitude exceeds threshold, scale or clamp gradients.
3. Then apply SGD update.

When to use:
- If loss becomes `NaN`, `Inf`, or jumps wildly.
- If generation collapses to repeated noise too early.

Acceptance criteria:
- Training runs for full epochs without numerical blow-up.

---

## Sprint 4: Inference (Generation)

### Task 10: Greedy Decoding

Objective: deterministically pick next character.

Steps:
1. Hardcode a seed prompt in `main.go` (example: `ROMEO:`).
2. Convert seed to IDs using vocab map.
3. Keep only the last `N=8` IDs as initial context.
4. Run forward pass to get logits.
5. Pick highest logit index (argmax).
6. Convert index to character and append to output text.

Acceptance criteria:
- One forward step returns one valid next character.

---

### Task 11: Generation Loop

Objective: generate long text by repeatedly sliding context.

Steps:
1. Repeat greedy decoding for `K` steps (start with 100).
2. After each predicted character:
   - append to output
   - update context window by dropping oldest ID and adding new ID
3. Print final generated text.

Acceptance criteria:
- Output length grows by exactly `K` characters.
- Program runs end-to-end with no runtime errors.

---

## Training Orchestration Checklist (Practical Run Order)

Use this exact implementation order:

1. Data loading and vocabulary creation
2. Corpus encoding
3. Dataset pair creation
4. Parameter initialization
5. Forward pass for one sample
6. Loss for one sample
7. Backward pass for one sample
8. SGD update for one sample
9. Batch processing
10. Epoch loop with average loss logging
11. Generation from hardcoded prompt

If stuck, always reduce complexity:
- One sample before batch
- One batch before full epoch
- One generation step before 100-step loop

---

## Recommended Logging During Development

Print these values periodically:
- `vocabSize`
- Number of training pairs
- Tensor shapes at each forward stage
- Batch loss and epoch average loss
- Whether any value becomes `NaN` or `Inf`

This catches 90% of beginner bugs quickly (especially shape mismatches).

---

## Common Pitfalls (and Fixes)

1. Shape mismatch in matrix multiplication
- Fix by writing expected dimensions next to each tensor and checking every layer.

2. Wrong target alignment
- Fix by validating a few decoded `(context → target)` pairs manually.

3. Unstable softmax / loss
- Fix with numerically stable softmax and gradient clipping.

4. Learning rate too high
- If loss explodes, reduce from `0.1` to `0.05` or `0.01`.

5. Prompt characters not in vocab
- Ensure prompt uses characters that appear in training text, or define unknown-char handling.

---

## Minimal “Definition of Done”

You are done when all are true:
- Program reads `data/input.txt` and builds char vocab.
- Program trains FNN for at least 10 epochs.
- Loss generally decreases across epochs.
- `main.go` contains a hardcoded prompt.
- Running the app prints generated continuation text (100+ characters).

---

## Suggested Milestones for One Weekend Build

Day 1:
- Finish Sprint 1 and verify data pipeline thoroughly.
- Complete forward pass (Sprint 2) with shape checks.

Day 2:
- Implement loss + backward + SGD + clipping.
- Run training and print epoch losses.
- Implement generation loop and get first coherent-looking output.

---

## What “Success” Looks Like

Your output will not be perfect English at first. Early success is:
- Character patterns from Shakespeare style appear (punctuation, spacing, names).
- Fewer random symbols over time.
- Loss trend is downward.

That is exactly the expected first milestone before moving to bigger architectures.
