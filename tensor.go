package main

import (
	"errors"
	"math"
	"runtime"
	"sync"
)

var numWorkers = runtime.NumCPU()

type Tensor struct {
	Data  []float64
	Shape []int
}

// MatMul performs matrix multiplication on 2D tensors with parallel computation
func MatMul(t1, t2 *Tensor) (*Tensor, error) {
	if len(t1.Shape) != 2 || len(t2.Shape) != 2 {
		return nil, errors.New("both tensors must be 2D")
	}

	rows1, cols1 := t1.Shape[0], t1.Shape[1]
	rows2, cols2 := t2.Shape[0], t2.Shape[1]

	if cols1 != rows2 {
		return nil, errors.New("incompatible shapes for multiplication")
	}

	result := make([]float64, rows1*cols2)

	// Parallelize row computation
	var wg sync.WaitGroup
	rowsPerWorker := (rows1 + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		startRow := w * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if endRow > rows1 {
			endRow = rows1
		}
		if startRow >= rows1 {
			break
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				for j := 0; j < cols2; j++ {
					sum := 0.0
					for k := 0; k < cols1; k++ {
						sum += t1.Data[i*cols1+k] * t2.Data[k*cols2+j]
					}
					result[i*cols2+j] = sum
				}
			}
		}(startRow, endRow)
	}
	wg.Wait()

	return &Tensor{Data: result, Shape: []int{rows1, cols2}}, nil
}

// Add performs element-wise addition with parallel computation
func Add(t1, t2 *Tensor) (*Tensor, error) {
	if len(t1.Data) != len(t2.Data) {
		return nil, errors.New("tensors must have same size")
	}

	result := make([]float64, len(t1.Data))
	parallelFor(len(result), func(start, end int) {
		for i := start; i < end; i++ {
			result[i] = t1.Data[i] + t2.Data[i]
		}
	})

	return &Tensor{Data: result, Shape: t1.Shape}, nil
}

// Lookup retrieves embedding vectors by token IDs
func Lookup(embeddings *Tensor, tokenIDs []int) (*Tensor, error) {
	if len(embeddings.Shape) != 2 {
		return nil, errors.New("embeddings must be 2D")
	}

	vocabSize, embedDim := embeddings.Shape[0], embeddings.Shape[1]
	result := make([]float64, len(tokenIDs)*embedDim)

	for i, id := range tokenIDs {
		if id < 0 || id >= vocabSize {
			return nil, errors.New("token ID out of range")
		}
		copy(result[i*embedDim:], embeddings.Data[id*embedDim:(id+1)*embedDim])
	}

	return &Tensor{Data: result, Shape: []int{len(tokenIDs), embedDim}}, nil
}

// Flatten converts tensor to 1D
func Flatten(t *Tensor) *Tensor {
	return &Tensor{Data: t.Data, Shape: []int{len(t.Data)}}
}

// ReLU applies max(0, x) element-wise with parallel computation
func ReLU(t *Tensor) *Tensor {
	result := make([]float64, len(t.Data))
	parallelFor(len(t.Data), func(start, end int) {
		for i := start; i < end; i++ {
			if t.Data[i] > 0 {
				result[i] = t.Data[i]
			}
		}
	})
	return &Tensor{Data: result, Shape: t.Shape}
}

func SoftMax(t *Tensor) (*Tensor, error) {
	if t == nil {
		return nil, errors.New("tensor cannot be nil")
	}
	if len(t.Data) == 0 {
		return nil, errors.New("tensor data cannot be empty")
	}

	resultData := make([]float64, len(t.Data))
	exaggerated := make([]float64, len(t.Data))

	maxVal := t.Data[0]
	for _, value := range t.Data[1:] {
		if value > maxVal {
			maxVal = value
		}
	}

	sum := 0.0
	for i, value := range t.Data {
		exaggerated[i] = math.Exp(value - maxVal)
		sum += exaggerated[i]
	}

	if sum == 0 || math.IsInf(sum, 0) || math.IsNaN(sum) {
		return nil, errors.New("invalid softmax denominator")
	}

	for i, value := range exaggerated {
		resultData[i] = value / sum
	}

	return &Tensor{
		Data:  resultData,
		Shape: t.Shape,
	}, nil
}

// ArgMax returns the index of the largest value in a tensor.
func ArgMax(t *Tensor) (int, error) {
	if t == nil {
		return -1, errors.New("tensor cannot be nil")
	}
	if len(t.Data) == 0 {
		return -1, errors.New("tensor data cannot be empty")
	}

	maxIndex := 0
	maxValue := t.Data[0]

	for i := 1; i < len(t.Data); i++ {
		if t.Data[i] > maxValue {
			maxValue = t.Data[i]
			maxIndex = i
		}
	}

	return maxIndex, nil
}

// AddBias adds bias vector to each row of a 2D tensor (broadcasting) with parallel computation
func AddBias(t *Tensor, bias *Tensor) (*Tensor, error) {
	if len(t.Shape) != 2 {
		return nil, errors.New("tensor must be 2D")
	}
	if len(bias.Shape) != 1 || bias.Shape[0] != t.Shape[1] {
		return nil, errors.New("bias shape must match tensor columns")
	}

	rows, cols := t.Shape[0], t.Shape[1]
	result := make([]float64, len(t.Data))

	parallelFor(rows, func(startRow, endRow int) {
		for i := startRow; i < endRow; i++ {
			for j := 0; j < cols; j++ {
				result[i*cols+j] = t.Data[i*cols+j] + bias.Data[j]
			}
		}
	})

	return &Tensor{Data: result, Shape: []int{rows, cols}}, nil
}

// Zeros creates a zero tensor with given shape
func Zeros(shape ...int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}
	return &Tensor{Data: make([]float64, size), Shape: shape}
}

// Clone creates a copy of the tensor
func (t *Tensor) Clone() *Tensor {
	data := make([]float64, len(t.Data))
	copy(data, t.Data)
	shape := make([]int, len(t.Shape))
	copy(shape, t.Shape)
	return &Tensor{Data: data, Shape: shape}
}

// Scale multiplies all elements by a scalar with parallel computation
func Scale(t *Tensor, s float64) *Tensor {
	result := make([]float64, len(t.Data))
	parallelFor(len(t.Data), func(start, end int) {
		for i := start; i < end; i++ {
			result[i] = t.Data[i] * s
		}
	})
	return &Tensor{Data: result, Shape: t.Shape}
}

// ClipNorm clips gradient norm to threshold with parallel norm computation
func ClipNorm(t *Tensor, threshold float64) *Tensor {
	// Parallel norm computation using partial sums
	partialSums := make([]float64, numWorkers)
	parallelForIdx(len(t.Data), func(workerIdx, start, end int) {
		sum := 0.0
		for i := start; i < end; i++ {
			sum += t.Data[i] * t.Data[i]
		}
		partialSums[workerIdx] = sum
	})

	norm := 0.0
	for _, ps := range partialSums {
		norm += ps
	}
	norm = math.Sqrt(norm)

	if norm > threshold {
		scale := threshold / norm
		return Scale(t, scale)
	}
	return t.Clone()
}

// parallelFor executes a function in parallel over a range [0, n)
func parallelFor(n int, fn func(start, end int)) {
	if n <= 0 {
		return
	}
	if n < numWorkers*4 {
		// Too small to parallelize effectively
		fn(0, n)
		return
	}

	var wg sync.WaitGroup
	chunkSize := (n + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > n {
			end = n
		}
		if start >= n {
			break
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			fn(s, e)
		}(start, end)
	}
	wg.Wait()
}

// parallelForIdx executes a function in parallel with worker index
func parallelForIdx(n int, fn func(workerIdx, start, end int)) {
	if n <= 0 {
		return
	}

	var wg sync.WaitGroup
	chunkSize := (n + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > n {
			end = n
		}
		if start >= n {
			break
		}

		wg.Add(1)
		go func(idx, s, e int) {
			defer wg.Done()
			fn(idx, s, e)
		}(w, start, end)
	}
	wg.Wait()
}
