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
	if model.PredictProbability(features) > DefaultClassThreshold {
		return StrokePositiveClass
	}
	return 0
}

// Train performs gradient descent on the training data.
func (model *LogisticModel) Train(data []DataPoint, epochs int, learningRate float64) {
	for range epochs {
		for _, dp := range data {
			prediction := model.PredictProbability(dp.Features)

			// difference between predicted probability and actual label, not an actual error
			err := prediction - float64(dp.Label)

			if dp.Label == 1 {
				err *= DefaultPenalty
			}

			for featureIndex, featureValue := range dp.Features {
				model.Weights[featureIndex] -= learningRate * err * featureValue
			}
			model.Bias -= learningRate * err
		}
	}
}

// Evaluate returns accuracy on test data.
func (model *LogisticModel) Evaluate(data []DataPoint) (float64, []int) {
	if len(data) == 0 {
		return 0.0, make([]int, 4)
	}

	tp := 0
	tn := 0
	fp := 0
	fn := 0

	correct := 0
	for _, dp := range data {
		if model.Predict(dp.Features) == dp.Label {
			correct++
			if dp.Label == StrokePositiveClass {
				tp++
			} else {
				tn++
			}
		} else {
			if dp.Label == StrokePositiveClass {
				fn++
			} else {
				fp++
			}
		}
	}

	return float64(correct) / float64(len(data)), []int{tp, tn, fp, fn}
}

// PrintWeights displays learned weights.
func (model *LogisticModel) PrintWeights() {
	fmt.Println("weights:")
	for featureIndex, featureName := range model.FeatureNames {
		fmt.Printf("  %s: %.4f\n", featureName, model.Weights[featureIndex])
	}
}
