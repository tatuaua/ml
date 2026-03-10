package main

import (
	"fmt"
	"math/rand"
	"time"
)

// ShuffleData returns a shuffled copy of the data.
func ShuffleData(data []DataPoint, rng *rand.Rand) []DataPoint {
	shuffled := make([]DataPoint, len(data))
	copy(shuffled, data)
	rng.Shuffle(len(shuffled), func(leftIndex, rightIndex int) {
		shuffled[leftIndex], shuffled[rightIndex] = shuffled[rightIndex], shuffled[leftIndex]
	})
	return shuffled
}

// TrainTestSplit splits data into train and test sets.
func TrainTestSplit(data []DataPoint, trainRatio float64) (train, test []DataPoint) {
	splitIdx := int(trainRatio * float64(len(data)))
	if splitIdx < 1 {
		splitIdx = 1
	}
	if splitIdx >= len(data) {
		splitIdx = len(data) - 1
	}
	return data[:splitIdx], data[splitIdx:]
}

// =============================================================================
// Main
// =============================================================================

func main() {
	// 1. Choose dataset configuration (swap this to use a different dataset)
	config := StrokeDatasetConfig()

	// 2. Load and parse data
	records, err := LoadCSV(DatasetCSVPath)
	if err != nil {
		panic(err)
	}

	data := LoadDataset(config, records)
	if len(data) < 2 {
		fmt.Println("not enough valid rows after parsing")
		return
	}

	// 3. Shuffle and split
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	shuffled := ShuffleData(data, rng)
	trainData, testData := TrainTestSplit(shuffled, DefaultTrainRatio)

	// 4. Fit scaler on training data and transform both sets
	scaler := FitScaler(trainData)
	scaler.Transform(trainData)
	scaler.Transform(testData)

	fmt.Printf("Dataset: %s\n", config.Name)
	fmt.Printf("Total valid rows: %d (train: %d, test: %d)\n", len(data), len(trainData), len(testData))

	// 5. Train model
	model := NewLogisticModel(config)
	model.Train(trainData, DefaultEpochs, DefaultLearningRate)

	// 6. Evaluate
	accuracy, confusion := model.Evaluate(testData)
	fmt.Printf("Test accuracy: %.2f%%\n", accuracy*100)
	fmt.Printf("Confusion matrix (rows=actual, cols=predicted):\n")
	fmt.Printf("                Pred 0  Pred 1\n")
	fmt.Printf("Actual 0 (TN/FP): %6d  %6d\n", confusion[1], confusion[2])
	fmt.Printf("Actual 1 (FN/TP): %6d  %6d\n", confusion[3], confusion[0])

	// Temporary hardcoded test row.
	hardcodedRow := HardcodedTestRow
	testConfig := config
	testConfig.SkipHeader = false
	hardcodedData := LoadDataset(testConfig, [][]string{hardcodedRow})
	if len(hardcodedData) == 1 {
		scaler.Transform(hardcodedData)
		prob := model.PredictProbability(hardcodedData[0].Features)
		fmt.Printf("%s: predicted=%.2f (guess=%d), actual=%d\n",
			HardcodedTestMessage,
			prob, model.Predict(hardcodedData[0].Features), hardcodedData[0].Label)
	} else {
		fmt.Printf("%s: parse failed\n", HardcodedTestMessage)
	}

	model.PrintWeights()
}
