package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

type HealthRow struct {
	ID              int
	Gender          string
	Age             float64
	Hypertension    int
	HeartDisease    int
	EverMarried     string
	WorkType        string
	ResidenceType   string
	AvgGlucoseLevel float64
	BMI             float64
	SmokingStatus   string
	Stroke          int
}

var healthRows []HealthRow

var featureNames = []string{"age", "hypertension", "heart_disease", "avg_glucose", "bmi"}

func featureValues(row HealthRow) []float64 {
	return []float64{
		row.Age,
		float64(row.Hypertension),
		float64(row.HeartDisease),
		row.AvgGlucoseLevel,
		row.BMI,
	}
}

func main() {
	data, err := os.ReadFile("../data/healthcare-dataset-stroke-data.csv")
	if err != nil {
		panic(err)
	}

	reader := csv.NewReader(strings.NewReader(string(data)))
	records, err := reader.ReadAll()
	if err != nil {
		panic(err)
	}

	for i, row := range records {
		if i == 0 {
			continue
		}
		if len(row) < 12 {
			continue
		}

		id, err := strconv.Atoi(row[0])
		if err != nil {
			continue
		}
		age, err := strconv.ParseFloat(row[2], 64)
		if err != nil {
			continue
		}
		hypertension, err := strconv.Atoi(row[3])
		if err != nil {
			continue
		}
		heartDisease, err := strconv.Atoi(row[4])
		if err != nil {
			continue
		}
		avgGlucose, err := strconv.ParseFloat(row[8], 64)
		if err != nil {
			continue
		}
		bmi, err := strconv.ParseFloat(row[9], 64)
		if err != nil {
			continue
		}
		stroke, err := strconv.Atoi(row[11])
		if err != nil {
			continue
		}

		healthRows = append(healthRows, HealthRow{
			ID:              id,
			Gender:          row[1],
			Age:             age,
			Hypertension:    hypertension,
			HeartDisease:    heartDisease,
			EverMarried:     row[5],
			WorkType:        row[6],
			ResidenceType:   row[7],
			AvgGlucoseLevel: avgGlucose,
			BMI:             bmi,
			SmokingStatus:   row[10],
			Stroke:          stroke,
		})
	}

	if len(healthRows) < 2 {
		fmt.Println("not enough valid rows after parsing")
		return
	}

	// Shuffle rows so train/test are randomly sampled instead of tail-based.
	shuffledRows := make([]HealthRow, len(healthRows))
	copy(shuffledRows, healthRows)
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	rng.Shuffle(len(shuffledRows), func(i, j int) {
		shuffledRows[i], shuffledRows[j] = shuffledRows[j], shuffledRows[i]
	})

	// Simple holdout: 80% train, 20% test.
	splitIdx := max(int(0.8*float64(len(shuffledRows))), 1)
	if splitIdx >= len(shuffledRows) {
		splitIdx = len(shuffledRows) - 1
	}

	trainRows := shuffledRows[:splitIdx]
	testRows := shuffledRows[splitIdx:]

	fmt.Printf("total valid rows: %d (train: %d, test: %d)\n", len(healthRows), len(trainRows), len(testRows))

	weights := make([]float64, len(featureNames))
	bias := 0.0
	learningRate := 0.1
	reps := 100
	predictProbability := func(row HealthRow) float64 {
		features := featureValues(row)
		score := bias
		for i, feature := range features {
			score += feature * weights[i]
		}
		return 1.0 / (1.0 + math.Exp(-score))
	}

	for reps >= 1 {
		for _, healthRow := range trainRows {
			features := featureValues(healthRow)
			prediction := predictProbability(healthRow)
			error := prediction - float64(healthRow.Stroke)
			for i, feature := range features {
				weights[i] -= learningRate * error * feature
			}
			bias -= learningRate * error
		}
		reps--
	}

	correct := 0
	for _, healthRow := range testRows {
		prediction := predictProbability(healthRow)
		guess := 0
		if prediction > 0.5 {
			guess = 1
		}
		if guess == healthRow.Stroke {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(testRows))
	prediction := predictProbability(testRows[0])
	guess := prediction > 0.5

	fmt.Printf("test accuracy: %.2f%%\n", accuracy*100)
	fmt.Printf("example test guess: %v, answer: %v\n", guess, testRows[0].Stroke)
	fmt.Println("weights:")
	for i, featureName := range featureNames {
		fmt.Printf("  %s: %.4f\n", featureName, weights[i])
	}
}
