package main

const (
	DatasetCSVPath       = "../data/healthcare-dataset-stroke-data.csv"
	DefaultTrainRatio    = 0.8
	DefaultEpochs        = 1000
	DefaultLearningRate  = 0.1
	HardcodedTestMessage = "Hardcoded row test"

	StrokeDatasetName      = "Stroke Prediction"
	StrokeLabelColumn      = 11
	StrokeMinColumns       = 12
	StrokePositiveClass    = 1
	DefaultClassThreshold  = 0.5
	ConfusionMatrixEntries = 4
)

var StrokeFeatureNames = []string{"age", "hypertension", "heart_disease", "avg_glucose", "bmi"}

var StrokeFeatureColumns = []int{2, 3, 4, 8, 9}

var HardcodedTestRow = []string{
	"71639",
	"Female",
	"68",
	"0",
	"0",
	"No",
	"Govt_job",
	"Urban",
	"82.1",
	"27.1",
	"Unknown",
	"1",
}
