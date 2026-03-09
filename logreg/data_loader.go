package main

import (
	"encoding/csv"
	"os"
	"strconv"
	"strings"
)

// LoadDataset parses CSV records into DataPoints using the given config.
func LoadDataset(config DatasetConfig, records [][]string) []DataPoint {
	var result []DataPoint

	startIdx := 0
	if config.SkipHeader {
		startIdx = 1
	}

	for rowIndex := startIdx; rowIndex < len(records); rowIndex++ {
		row := records[rowIndex]
		if len(row) < config.MinColumns {
			continue
		}

		features := make([]float64, config.NumFeatures())
		valid := true
		for featureIndex, colIdx := range config.FeatureColumns {
			val, err := strconv.ParseFloat(row[colIdx], 64)
			if err != nil {
				valid = false
				break
			}
			features[featureIndex] = val
		}
		if !valid {
			continue
		}

		label, err := strconv.Atoi(row[config.LabelColumn])
		if err != nil {
			continue
		}

		result = append(result, DataPoint{
			Features: features,
			Label:    label,
		})
	}

	return result
}

// LoadCSV reads and parses a CSV file.
func LoadCSV(path string) ([][]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	reader := csv.NewReader(strings.NewReader(string(data)))
	return reader.ReadAll()
}
