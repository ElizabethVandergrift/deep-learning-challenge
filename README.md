# Alphabet Soup Funding Prediction

This project aims to build a neural network model to predict the success of funding applicants for Alphabet Soup.

## Overview of the Analysis

The purpose of this analysis was to develop a predictive model using a deep learning neural network. The goal was to accurately classify whether applicants would be successful in securing funding based on features from their application data.

## Data Preprocessing

### Target Variable
- **Target Variable**: `IS_SUCCESSFUL`, which indicates whether the applicant was successful in securing funding.

### Feature Variables
- **Feature Variables**: The features used in the model include:
  - **Categorical Features** (one-hot encoded):
    - `APPLICATION_TYPE`
    - `AFFILIATION`
    - `CLASSIFICATION`
    - `USE_CASE`
    - `ORGANIZATION`
    - `STATUS`
    - `INCOME_AMT`
    - `SPECIAL_CONSIDERATIONS`
  - **Continuous Feature**:
    - `ASK_AMT`: The dollar amount requested by the applicant, scaled using Min-Max Scaling.

### Variables Removed
- **Removed Variables**: 
  - `EIN` and `NAME` were removed from the dataset because they are unique identifiers and do not provide predictive power for the model.

## Model Architecture

The neural network was structured with the following architecture to capture complex relationships in the data:

```python
# Updated Model Architecture
nn = Sequential()
nn.add(Input(shape=(45,)))
nn.add(Dense(units=1512, activation="relu"))
nn.add(Dropout(0.3))  # Adding Dropout to prevent overfitting
nn.add(Dense(units=512, activation="relu"))
nn.add(Dropout(0.3))  # Adding Dropout to prevent overfitting
nn.add(Dense(units=256, activation="relu"))
nn.add(Dense(units=1, activation="sigmoid"))
````

## Neural Network Architecture

### Increased Units and Layers
The neural network architecture was adjusted to include more neurons and additional layers to better capture complex relationships in the data:

- **1st Hidden Layer**: 1512 units with ReLU activation
- **2nd Hidden Layer**: 512 units with ReLU activation
- **3rd Hidden Layer**: 256 units with ReLU activation
- **Output Layer**: 1 unit with Sigmoid activation
- **Dropout Layers**: Dropout was added after the first and second hidden layers to prevent overfitting, with a dropout rate of 0.3.

### Optimizers

#### Optimizer Experimentation
Various optimizers were tested to find the best performing one:

- **Adam**: Standard choice, often used as a baseline.
- **RMSprop**: Handles noisy and non-stationary data well.
- **SGD**: Simple and effective, especially with momentum.
- **Adamax**: A variant of Adam, which sometimes performs better on specific datasets.

#### Best Results
The Adamax optimizer was found to deliver the best results, with a learning rate of 0.002:

- **Adamax Optimizer**:
  - **Loss**: 0.5467
  - **Accuracy**: 73.17%

### Summary
The adjusted neural network model with increased units and layers, combined with the Adamax optimizer, provided the best performance in predicting the success of funding applicants. Although the accuracy of 73.17% was slightly below the target of 75%, it represents a significant improvement over previous iterations.

### Recommendation for Improvement
- **Alternative Models**: Consider exploring tree-based models like Random Forest or XGBoost, which may handle the categorical and continuous mix more effectively.
