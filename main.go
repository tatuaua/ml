package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"sync"
	"time"
)

// Hyperparameters
const (
	contextSize  = 8
	embedDim     = 10
	hiddenDim    = 128
	batchSize    = 32
	learningRate = 0.1
	epochs       = 10
	clipThresh   = 5.0
	genLength    = 100
)

// Vocabulary holds character-to-ID mappings
type Vocabulary struct {
	CharToID map[rune]int
	IDToChar map[int]rune
	Size     int
}

// BuildVocabulary creates vocabulary from text
func BuildVocabulary(text string) *Vocabulary {
	// Collect unique characters
	charSet := make(map[rune]bool)
	for _, ch := range text {
		charSet[ch] = true
	}

	// Sort for deterministic ordering
	chars := make([]rune, 0, len(charSet))
	for ch := range charSet {
		chars = append(chars, ch)
	}
	sort.Slice(chars, func(i, j int) bool { return chars[i] < chars[j] })

	// Build mappings
	charToID := make(map[rune]int)
	idToChar := make(map[int]rune)
	for i, ch := range chars {
		charToID[ch] = i
		idToChar[i] = ch
	}

	return &Vocabulary{
		CharToID: charToID,
		IDToChar: idToChar,
		Size:     len(chars),
	}
}

// Encode converts string to token IDs
func (v *Vocabulary) Encode(text string) []int {
	ids := make([]int, 0, len(text))
	for _, ch := range text {
		if id, ok := v.CharToID[ch]; ok {
			ids = append(ids, id)
		}
	}
	return ids
}

// Decode converts token IDs to string
func (v *Vocabulary) Decode(ids []int) string {
	chars := make([]rune, len(ids))
	for i, id := range ids {
		chars[i] = v.IDToChar[id]
	}
	return string(chars)
}

// Dataset holds training samples
type Dataset struct {
	Contexts [][]int // Each context has N token IDs
	Targets  []int   // Each target is next token ID
}

// BuildDataset creates sliding window training pairs
func BuildDataset(encoded []int, n int) *Dataset {
	numSamples := len(encoded) - n
	contexts := make([][]int, numSamples)
	targets := make([]int, numSamples)

	for i := 0; i < numSamples; i++ {
		ctx := make([]int, n)
		copy(ctx, encoded[i:i+n])
		contexts[i] = ctx
		targets[i] = encoded[i+n]
	}

	return &Dataset{Contexts: contexts, Targets: targets}
}

// Shuffle randomizes the dataset order
func (d *Dataset) Shuffle() {
	for i := len(d.Contexts) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		d.Contexts[i], d.Contexts[j] = d.Contexts[j], d.Contexts[i]
		d.Targets[i], d.Targets[j] = d.Targets[j], d.Targets[i]
	}
}

// Model represents the FNN
type Model struct {
	Embeddings *Tensor // (vocabSize, embedDim)
	W1         *Tensor // (contextSize*embedDim, hiddenDim)
	B1         *Tensor // (hiddenDim)
	W2         *Tensor // (hiddenDim, vocabSize)
	B2         *Tensor // (vocabSize)
}

// NewModel initializes model with random weights
func NewModel(vocabSize int) *Model {
	inputDim := contextSize * embedDim

	// Xavier initialization
	embScale := 0.1
	w1Scale := math.Sqrt(2.0 / float64(inputDim))
	w2Scale := math.Sqrt(2.0 / float64(hiddenDim))

	embeddings := make([]float64, vocabSize*embedDim)
	for i := range embeddings {
		embeddings[i] = (rand.Float64()*2 - 1) * embScale
	}

	w1 := make([]float64, inputDim*hiddenDim)
	for i := range w1 {
		w1[i] = (rand.Float64()*2 - 1) * w1Scale
	}

	b1 := make([]float64, hiddenDim)

	w2 := make([]float64, hiddenDim*vocabSize)
	for i := range w2 {
		w2[i] = (rand.Float64()*2 - 1) * w2Scale
	}

	b2 := make([]float64, vocabSize)

	return &Model{
		Embeddings: &Tensor{Data: embeddings, Shape: []int{vocabSize, embedDim}},
		W1:         &Tensor{Data: w1, Shape: []int{inputDim, hiddenDim}},
		B1:         &Tensor{Data: b1, Shape: []int{hiddenDim}},
		W2:         &Tensor{Data: w2, Shape: []int{hiddenDim, vocabSize}},
		B2:         &Tensor{Data: b2, Shape: []int{vocabSize}},
	}
}

// ForwardCache stores intermediate values for backprop
type ForwardCache struct {
	TokenIDs   [][]int  // (B, N)
	Embedded   *Tensor  // (B, N*embedDim)
	HiddenPre  *Tensor  // (B, hiddenDim) pre-ReLU
	HiddenPost *Tensor  // (B, hiddenDim) post-ReLU
	Logits     *Tensor  // (B, vocabSize)
	Probs      *Tensor  // (B, vocabSize)
}

// Forward computes model output for a batch with parallel processing
func (m *Model) Forward(contexts [][]int) (*ForwardCache, error) {
	batchLen := len(contexts)
	inputDim := contextSize * embedDim
	vocabSize := m.Embeddings.Shape[0]
	numWorkers := runtime.NumCPU()

	// Embed and flatten each context in parallel
	embedded := make([]float64, batchLen*inputDim)
	var embErr error
	var errMu sync.Mutex
	var wg sync.WaitGroup

	chunkSize := (batchLen + numWorkers - 1) / numWorkers
	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > batchLen {
			end = batchLen
		}
		if start >= batchLen {
			break
		}

		wg.Add(1)
		go func(startB, endB int) {
			defer wg.Done()
			for b := startB; b < endB; b++ {
				emb, err := Lookup(m.Embeddings, contexts[b])
				if err != nil {
					errMu.Lock()
					if embErr == nil {
						embErr = err
					}
					errMu.Unlock()
					return
				}
				flat := Flatten(emb)
				copy(embedded[b*inputDim:], flat.Data)
			}
		}(start, end)
	}
	wg.Wait()
	if embErr != nil {
		return nil, embErr
	}
	embTensor := &Tensor{Data: embedded, Shape: []int{batchLen, inputDim}}

	// Hidden layer: h_pre = x @ W1 + b1
	hiddenPre, err := MatMul(embTensor, m.W1)
	if err != nil {
		return nil, err
	}
	hiddenPre, err = AddBias(hiddenPre, m.B1)
	if err != nil {
		return nil, err
	}

	// ReLU activation
	hiddenPost := ReLU(hiddenPre)

	// Output layer: logits = h @ W2 + b2
	logits, err := MatMul(hiddenPost, m.W2)
	if err != nil {
		return nil, err
	}
	logits, err = AddBias(logits, m.B2)
	if err != nil {
		return nil, err
	}

	// Softmax per row in parallel
	probs := make([]float64, batchLen*vocabSize)
	var softmaxErr error

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > batchLen {
			end = batchLen
		}
		if start >= batchLen {
			break
		}

		wg.Add(1)
		go func(startB, endB int) {
			defer wg.Done()
			for b := startB; b < endB; b++ {
				rowStart := b * vocabSize
				rowEnd := rowStart + vocabSize
				rowTensor := &Tensor{Data: logits.Data[rowStart:rowEnd], Shape: []int{vocabSize}}
				softmax, err := SoftMax(rowTensor)
				if err != nil {
					errMu.Lock()
					if softmaxErr == nil {
						softmaxErr = err
					}
					errMu.Unlock()
					return
				}
				copy(probs[rowStart:], softmax.Data)
			}
		}(start, end)
	}
	wg.Wait()
	if softmaxErr != nil {
		return nil, softmaxErr
	}
	probsTensor := &Tensor{Data: probs, Shape: []int{batchLen, vocabSize}}

	return &ForwardCache{
		TokenIDs:   contexts,
		Embedded:   embTensor,
		HiddenPre:  hiddenPre,
		HiddenPost: hiddenPost,
		Logits:     logits,
		Probs:      probsTensor,
	}, nil
}

// CrossEntropyLoss computes average loss over batch
func CrossEntropyLoss(probs *Tensor, targets []int) float64 {
	batchLen := len(targets)
	vocabSize := probs.Shape[1]
	totalLoss := 0.0

	for b := 0; b < batchLen; b++ {
		targetID := targets[b]
		prob := probs.Data[b*vocabSize+targetID]
		if prob < 1e-10 {
			prob = 1e-10 // Prevent log(0)
		}
		totalLoss -= math.Log(prob)
	}

	return totalLoss / float64(batchLen)
}

// Gradients holds parameter gradients
type Gradients struct {
	DEmbeddings *Tensor
	DW1         *Tensor
	DB1         *Tensor
	DW2         *Tensor
	DB2         *Tensor
}

// Backward computes gradients with parallel processing
func (m *Model) Backward(cache *ForwardCache, targets []int) *Gradients {
	batchLen := len(targets)
	vocabSize := m.Embeddings.Shape[0]
	inputDim := contextSize * embedDim
	numWorkers := runtime.NumCPU()

	// dLogits = probs - oneHot(targets) / batchSize (parallel)
	dLogits := make([]float64, batchLen*vocabSize)
	var wg sync.WaitGroup
	chunkSize := (batchLen + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > batchLen {
			end = batchLen
		}
		if start >= batchLen {
			break
		}

		wg.Add(1)
		go func(startB, endB int) {
			defer wg.Done()
			for b := startB; b < endB; b++ {
				for j := 0; j < vocabSize; j++ {
					idx := b*vocabSize + j
					dLogits[idx] = cache.Probs.Data[idx]
					if j == targets[b] {
						dLogits[idx] -= 1.0
					}
					dLogits[idx] /= float64(batchLen)
				}
			}
		}(start, end)
	}
	wg.Wait()
	dLogitsTensor := &Tensor{Data: dLogits, Shape: []int{batchLen, vocabSize}}

	// dW2 = hidden^T @ dLogits
	hiddenT := transpose(cache.HiddenPost)
	dW2, _ := MatMul(hiddenT, dLogitsTensor)

	// dB2 = sum(dLogits, axis=0) - parallel reduction
	dB2Partials := make([][]float64, numWorkers)
	for i := range dB2Partials {
		dB2Partials[i] = make([]float64, vocabSize)
	}

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > batchLen {
			end = batchLen
		}
		if start >= batchLen {
			break
		}

		wg.Add(1)
		go func(workerIdx, startB, endB int) {
			defer wg.Done()
			local := dB2Partials[workerIdx]
			for b := startB; b < endB; b++ {
				for j := 0; j < vocabSize; j++ {
					local[j] += dLogits[b*vocabSize+j]
				}
			}
		}(w, start, end)
	}
	wg.Wait()

	dB2 := make([]float64, vocabSize)
	for _, partial := range dB2Partials {
		for j := 0; j < vocabSize; j++ {
			dB2[j] += partial[j]
		}
	}

	// dHidden = dLogits @ W2^T
	w2T := transpose(m.W2)
	dHiddenPost, _ := MatMul(dLogitsTensor, w2T)

	// dHiddenPre = dHiddenPost * (hiddenPre > 0) [ReLU derivative] - parallel
	dHiddenPre := make([]float64, batchLen*hiddenDim)
	parallelFor(len(dHiddenPre), func(start, end int) {
		for i := start; i < end; i++ {
			if cache.HiddenPre.Data[i] > 0 {
				dHiddenPre[i] = dHiddenPost.Data[i]
			}
		}
	})
	dHiddenPreTensor := &Tensor{Data: dHiddenPre, Shape: []int{batchLen, hiddenDim}}

	// dW1 = embedded^T @ dHiddenPre
	embeddedT := transpose(cache.Embedded)
	dW1, _ := MatMul(embeddedT, dHiddenPreTensor)

	// dB1 = sum(dHiddenPre, axis=0) - parallel reduction
	dB1Partials := make([][]float64, numWorkers)
	for i := range dB1Partials {
		dB1Partials[i] = make([]float64, hiddenDim)
	}

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > batchLen {
			end = batchLen
		}
		if start >= batchLen {
			break
		}

		wg.Add(1)
		go func(workerIdx, startB, endB int) {
			defer wg.Done()
			local := dB1Partials[workerIdx]
			for b := startB; b < endB; b++ {
				for j := 0; j < hiddenDim; j++ {
					local[j] += dHiddenPre[b*hiddenDim+j]
				}
			}
		}(w, start, end)
	}
	wg.Wait()

	dB1 := make([]float64, hiddenDim)
	for _, partial := range dB1Partials {
		for j := 0; j < hiddenDim; j++ {
			dB1[j] += partial[j]
		}
	}

	// dEmbedded = dHiddenPre @ W1^T
	w1T := transpose(m.W1)
	dEmbedded, _ := MatMul(dHiddenPreTensor, w1T)

	// Accumulate embedding gradients - parallel with per-worker accumulators
	dEmbPartials := make([]*Tensor, numWorkers)
	for i := range dEmbPartials {
		dEmbPartials[i] = Zeros(vocabSize, embedDim)
	}

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > batchLen {
			end = batchLen
		}
		if start >= batchLen {
			break
		}

		wg.Add(1)
		go func(workerIdx, startB, endB int) {
			defer wg.Done()
			local := dEmbPartials[workerIdx]
			for b := startB; b < endB; b++ {
				for pos := 0; pos < contextSize; pos++ {
					tokenID := cache.TokenIDs[b][pos]
					for d := 0; d < embedDim; d++ {
						srcIdx := b*inputDim + pos*embedDim + d
						dstIdx := tokenID*embedDim + d
						local.Data[dstIdx] += dEmbedded.Data[srcIdx]
					}
				}
			}
		}(w, start, end)
	}
	wg.Wait()

	// Merge partial embedding gradients
	dEmbeddings := Zeros(vocabSize, embedDim)
	for _, partial := range dEmbPartials {
		for i := range dEmbeddings.Data {
			dEmbeddings.Data[i] += partial.Data[i]
		}
	}

	return &Gradients{
		DEmbeddings: dEmbeddings,
		DW1:         dW1,
		DB1:         &Tensor{Data: dB1, Shape: []int{hiddenDim}},
		DW2:         dW2,
		DB2:         &Tensor{Data: dB2, Shape: []int{vocabSize}},
	}
}

// transpose transposes a 2D tensor
func transpose(t *Tensor) *Tensor {
	rows, cols := t.Shape[0], t.Shape[1]
	result := make([]float64, len(t.Data))
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[j*rows+i] = t.Data[i*cols+j]
		}
	}
	return &Tensor{Data: result, Shape: []int{cols, rows}}
}

// SGDUpdate applies gradient descent with clipping
func (m *Model) SGDUpdate(grads *Gradients, lr float64) {
	// Clip gradients
	grads.DEmbeddings = ClipNorm(grads.DEmbeddings, clipThresh)
	grads.DW1 = ClipNorm(grads.DW1, clipThresh)
	grads.DB1 = ClipNorm(grads.DB1, clipThresh)
	grads.DW2 = ClipNorm(grads.DW2, clipThresh)
	grads.DB2 = ClipNorm(grads.DB2, clipThresh)

	// Update parameters
	for i := range m.Embeddings.Data {
		m.Embeddings.Data[i] -= lr * grads.DEmbeddings.Data[i]
	}
	for i := range m.W1.Data {
		m.W1.Data[i] -= lr * grads.DW1.Data[i]
	}
	for i := range m.B1.Data {
		m.B1.Data[i] -= lr * grads.DB1.Data[i]
	}
	for i := range m.W2.Data {
		m.W2.Data[i] -= lr * grads.DW2.Data[i]
	}
	for i := range m.B2.Data {
		m.B2.Data[i] -= lr * grads.DB2.Data[i]
	}
}

// Generate produces text from a prompt
func (m *Model) Generate(vocab *Vocabulary, prompt string, length int) string {
	// Encode prompt and take last N characters
	ids := vocab.Encode(prompt)
	if len(ids) < contextSize {
		// Pad with first character if needed
		padding := make([]int, contextSize-len(ids))
		ids = append(padding, ids...)
	} else if len(ids) > contextSize {
		ids = ids[len(ids)-contextSize:]
	}

	result := prompt
	ctx := make([]int, contextSize)
	copy(ctx, ids)

	for i := 0; i < length; i++ {
		// Forward pass with batch size 1
		cache, err := m.Forward([][]int{ctx})
		if err != nil {
			break
		}

		// Greedy decode: pick highest probability
		nextID, _ := ArgMax(&Tensor{Data: cache.Probs.Data, Shape: []int{len(cache.Probs.Data)}})

		// Append to result
		nextChar := vocab.IDToChar[nextID]
		result += string(nextChar)

		// Slide context window
		copy(ctx, ctx[1:])
		ctx[contextSize-1] = nextID
	}

	return result
}

func main() {
	// Load training data
	data, err := os.ReadFile("data/input.txt")
	if err != nil {
		fmt.Println("Error loading data:", err)
		return
	}
	text := string(data)

	// Build vocabulary
	vocab := BuildVocabulary(text)
	fmt.Printf("Vocabulary size: %d\n", vocab.Size)

	// Encode text
	encoded := vocab.Encode(text)
	fmt.Printf("Encoded length: %d\n", len(encoded))

	// Build dataset
	dataset := BuildDataset(encoded, contextSize)
	fmt.Printf("Training samples: %d\n", len(dataset.Targets))

	// Initialize model
	model := NewModel(vocab.Size)
	fmt.Printf("Model initialized (input: %d, hidden: %d, output: %d)\n",
		contextSize*embedDim, hiddenDim, vocab.Size)

	// Training loop
	numBatches := len(dataset.Targets) / batchSize
	fmt.Printf("\nTraining for %d epochs, %d batches per epoch\n\n", epochs, numBatches)

	logInterval := numBatches / 10 // Log 10 times per epoch
	if logInterval < 1 {
		logInterval = 1
	}

	trainingStart := time.Now()

	for epoch := 0; epoch < epochs; epoch++ {
		epochStart := time.Now()
		dataset.Shuffle()
		epochLoss := 0.0

		for b := 0; b < numBatches; b++ {
			// Get batch
			startIdx := b * batchSize
			batchContexts := dataset.Contexts[startIdx : startIdx+batchSize]
			batchTargets := dataset.Targets[startIdx : startIdx+batchSize]

			// Forward pass
			cache, err := model.Forward(batchContexts)
			if err != nil {
				fmt.Printf("Forward error: %v\n", err)
				return
			}

			// Compute loss
			loss := CrossEntropyLoss(cache.Probs, batchTargets)
			if math.IsNaN(loss) || math.IsInf(loss, 0) {
				fmt.Println("Loss became NaN/Inf, stopping training")
				return
			}
			epochLoss += loss

			// Backward pass
			grads := model.Backward(cache, batchTargets)

			// Update parameters
			model.SGDUpdate(grads, learningRate)

			// Progress logging
			if (b+1)%logInterval == 0 {
				fmt.Printf("  Epoch %d - Batch %d/%d (%.1f%%) - Loss: %.4f\n",
					epoch+1, b+1, numBatches, float64(b+1)/float64(numBatches)*100, loss)
			}
		}

		avgLoss := epochLoss / float64(numBatches)
		epochDuration := time.Since(epochStart)
		fmt.Printf("Epoch %d/%d - Loss: %.4f - Time: %v\n", epoch+1, epochs, avgLoss, epochDuration)
	}

	totalDuration := time.Since(trainingStart)
	fmt.Printf("\nTraining completed in %v\n", totalDuration)

	// Generation
	prompt := "ROMEO:"
	fmt.Printf("\n--- Generation ---\nPrompt: %s\n\n", prompt)
	generated := model.Generate(vocab, prompt, genLength)
	fmt.Println(generated)
}
