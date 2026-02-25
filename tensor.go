package main

import (
	"errors"
	"math"
)

type Tensor struct {
	Data  []float64
	Shape []int
}

// MatMul performs matrix multiplication on 2D tensors
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
	for i := 0; i < rows1; i++ {
		for j := 0; j < cols2; j++ {
			sum := 0.0
			for k := 0; k < cols1; k++ {
				sum += t1.Data[i*cols1+k] * t2.Data[k*cols2+j]
			}
			result[i*cols2+j] = sum
		}
	}

	return &Tensor{Data: result, Shape: []int{rows1, cols2}}, nil
}

// Add performs element-wise addition
func Add(t1, t2 *Tensor) (*Tensor, error) {
	if len(t1.Data) != len(t2.Data) {
		return nil, errors.New("tensors must have same size")
	}

	result := make([]float64, len(t1.Data))
	for i := range result {
		result[i] = t1.Data[i] + t2.Data[i]
	}

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

// ReLU applies max(0, x) element-wise
func ReLU(t *Tensor) *Tensor {
	result := make([]float64, len(t.Data))
	for i, v := range t.Data {
		if v > 0 {
			result[i] = v
		}
	}
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
