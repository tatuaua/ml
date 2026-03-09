package main

// DataPoint represents a single training/test example with features and label.
type DataPoint struct {
	Features []float64
	Label    int
}

// DatasetConfig defines how to parse a CSV into DataPoints.
// This is the single source of truth for feature definitions.
type DatasetConfig struct {
	Name           string
	FeatureNames   []string // Names for each feature (for display)
	FeatureColumns []int    // CSV column indices for each feature
	LabelColumn    int      // CSV column index for the label
	SkipHeader     bool     // Whether to skip the first row
	MinColumns     int      // Minimum columns required per row
}

// NumFeatures returns the number of features in this config.
func (config DatasetConfig) NumFeatures() int {
	return len(config.FeatureNames)
}

// StrokeDatasetConfig returns the config for the healthcare stroke dataset.
func StrokeDatasetConfig() DatasetConfig {
	return DatasetConfig{
		Name:           "Stroke Prediction",
		FeatureNames:   []string{"age", "hypertension", "heart_disease", "avg_glucose", "bmi"},
		FeatureColumns: []int{2, 3, 4, 8, 9},
		LabelColumn:    11,
		SkipHeader:     true,
		MinColumns:     12,
	}
}
