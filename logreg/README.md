# Logistic regression

## What

- Uses age, hypertension, heart disease, average glucose, and BMI to estimate how likely a stroke is in this dataset, then marks each case as `0` (no stroke) or `1` (stroke).

## How

- Splits rows of data randomly to training and test data
- Scales features (eg. age) to less chaotic numbers so they dont ruin the math
- Calculates bias and weights of each feature
- Calculates a score for each row of data in the test set to evaluate precision

## Cool words

- `min-max`: scaling that maps feature values to a 0-1 range. Used in training.
- `weights`: per-feature numbers the model learns.
- `bias`: extra offset value added to each score.
- `sigmoid`: function that converts a score into a value between 0 and 1. Not used in training.
- `epochs`: number of full passes through the training data.
- `learningRate`: step size used when updating `weights` and `bias`.
