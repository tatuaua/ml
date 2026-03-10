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

## Math

### Min-max scaling

Normalizes features to a 0-1 range:

$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

### Sigmoid function

Converts a score into a probability between 0 and 1:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### Linear combination (score)

Computes the raw score from features, weights, and bias:

$$z = b + \sum_{i=1}^{n} w_i \cdot x_i$$

### Predicted probability

Probability that the output is class 1:

$$P(y=1|x) = \sigma\left(b + \sum_{i=1}^{n} w_i \cdot x_i\right)$$

### Gradient descent update

Updates weights and bias each iteration:

$$w_i \leftarrow w_i - \alpha \cdot (prediction - label) \cdot x_i$$

$$b \leftarrow b - \alpha \cdot (prediction - label)$$

### Classification

Final prediction based on threshold (default 0.5):

$$\hat{y} = \begin{cases} 1 & \text{if } P(y=1|x) > 0.5 \\ 0 & \text{otherwise} \end{cases}$$

### Accuracy

$$accuracy = \frac{\text{correct predictions}}{\text{total predictions}}$$
