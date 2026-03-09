package main

import (
	"fmt"
	"math"
)

// LogisticModel encapsulates a trained logistic regression model.
type LogisticModel struct {
	Weights      []float64
	Bias         float64
	FeatureNames []string
}

// NewLogisticModel creates a new model with zeroed weights.
func NewLogisticModel(config DatasetConfig) *LogisticModel {
	return &LogisticModel{
		Weights:      make([]float64, config.NumFeatures()),
		Bias:         0.0,
		FeatureNames: config.FeatureNames,
	}
}

func sigmoid(input float64) float64 {
	return 1.0 / (1.0 + math.Exp(-input))
}

// PredictProbability returns P(label=1) for given features.
func (model *LogisticModel) PredictProbability(features []float64) float64 {
	score := model.Bias
	for featureIndex, featureValue := range features {
		score += featureValue * model.Weights[featureIndex]
	}
	return sigmoid(score)
}

// Predict returns 0 or 1 based on threshold (default 0.5).
func (model *LogisticModel) Predict(features []float64) int {
	if model.PredictProbability(features) > 0.5 {
		return 1
	}
	return 0
}

// Train performs gradient descent on the training data.
func (model *LogisticModel) Train(data []DataPoint, epochs int, learningRate float64) {
	for range epochs {
		for _, dp := range data {
			prediction := model.PredictProbability(dp.Features)
			err := prediction - float64(dp.Label)

			for featureIndex, featureValue := range dp.Features {
				model.Weights[featureIndex] -= learningRate * err * featureValue
			}
			model.Bias -= learningRate * err
		}
	}
}

// Evaluate returns accuracy on test data.
func (model *LogisticModel) Evaluate(data []DataPoint) float64 {
	if len(data) == 0 {
		return 0.0
	}

	correct := 0
	for _, dp := range data {
		if model.Predict(dp.Features) == dp.Label {
			correct++
		}
	}
	return float64(correct) / float64(len(data))
}

// PrintWeights displays learned weights.
func (model *LogisticModel) PrintWeights() {
	fmt.Println("weights:")
	for featureIndex, featureName := range model.FeatureNames {
		fmt.Printf("  %s: %.4f\n", featureName, model.Weights[featureIndex])
	}
}
