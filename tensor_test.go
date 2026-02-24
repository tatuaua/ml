package main

import (
	"testing"
)

func TestMatMul(t *testing.T) {
	t1 := &Tensor{Data: []float64{1, 2, 3, 4}, Shape: []int{2, 2}}
	t2 := &Tensor{Data: []float64{5, 6, 7, 8}, Shape: []int{2, 2}}

	result, err := MatMul(t1, t2)
	if err != nil {
		t.Fatal(err)
	}

	expected := []float64{19, 22, 43, 50}
	for i, v := range expected {
		if result.Data[i] != v {
			t.Errorf("index %d: expected %f, got %f", i, v, result.Data[i])
		}
	}
}

func TestAdd(t *testing.T) {
	t1 := &Tensor{Data: []float64{1, 2, 3}, Shape: []int{3}}
	t2 := &Tensor{Data: []float64{4, 5, 6}, Shape: []int{3}}

	result, err := Add(t1, t2)
	if err != nil {
		t.Fatal(err)
	}

	expected := []float64{5, 7, 9}
	for i, v := range expected {
		if result.Data[i] != v {
			t.Errorf("index %d: expected %f, got %f", i, v, result.Data[i])
		}
	}
}

func TestLookup(t *testing.T) {
	embeddings := &Tensor{
		Data:  []float64{1, 2, 3, 4, 5, 6, 7, 8},
		Shape: []int{4, 2},
	}

	result, err := Lookup(embeddings, []int{0, 2})
	if err != nil {
		t.Fatal(err)
	}

	expected := []float64{1, 2, 5, 6}
	for i, v := range expected {
		if result.Data[i] != v {
			t.Errorf("index %d: expected %f, got %f", i, v, result.Data[i])
		}
	}
}

func TestReLU(t *testing.T) {
	tensor := &Tensor{Data: []float64{-2, -1, 0, 1, 2}, Shape: []int{5}}
	result := ReLU(tensor)

	expected := []float64{0, 0, 0, 1, 2}
	for i, v := range expected {
		if result.Data[i] != v {
			t.Errorf("index %d: expected %f, got %f", i, v, result.Data[i])
		}
	}
}

func TestFlatten(t *testing.T) {
	tensor := &Tensor{Data: []float64{1, 2, 3, 4}, Shape: []int{2, 2}}
	result := Flatten(tensor)

	if len(result.Shape) != 1 || result.Shape[0] != 4 {
		t.Errorf("expected shape [4], got %v", result.Shape)
	}
}
