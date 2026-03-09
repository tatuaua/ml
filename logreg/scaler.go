package main

import "math"

// Scaler holds min-max normalization parameters.
type Scaler struct {
	Mins  []float64
	Maxes []float64
}

// FitScaler computes min/max values from training data.
func FitScaler(data []DataPoint) Scaler {
	if len(data) == 0 {
		return Scaler{}
	}

	numFeatures := len(data[0].Features)
	mins := make([]float64, numFeatures)
	maxes := make([]float64, numFeatures)

	for featureIndex := range mins {
		mins[featureIndex] = math.Inf(1)
		maxes[featureIndex] = math.Inf(-1)
	}

	for _, dp := range data {
		for featureIndex, val := range dp.Features {
			mins[featureIndex] = math.Min(mins[featureIndex], val)
			maxes[featureIndex] = math.Max(maxes[featureIndex], val)
		}
	}

	return Scaler{Mins: mins, Maxes: maxes}
}

// Transform applies min-max scaling to data points in place.
func (scaler Scaler) Transform(data []DataPoint) {
	for dataIndex := range data {
		for featureIndex := range data[dataIndex].Features {
			denom := scaler.Maxes[featureIndex] - scaler.Mins[featureIndex]
			if denom > 0 {
				data[dataIndex].Features[featureIndex] = (data[dataIndex].Features[featureIndex] - scaler.Mins[featureIndex]) / denom
			}
		}
	}
}
