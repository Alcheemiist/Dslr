
import sys
import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


class LogisticRegressionSGD:
    def __init__(self, weights):
        self.weights = weights

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        # Add bias term
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return self.sigmoid(np.dot(X, self.weights)) >= 0.5


def predict_multi_class_prob(models, X):
    # Get the probability scores instead of boolean predictions
    scores = {house: model.sigmoid(np.dot(np.hstack([np.ones((X.shape[0], 1)), X]), model.weights))
              for house, model in models.items()}
    return pd.DataFrame(scores).idxmax(axis=1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python logreg_predict.py <test_data.csv>")
        exit(1)

    # Load test data
    test_data = pd.read_csv(sys.argv[1])

    # Extract true labels and features
    X_test = test_data.drop(
        columns=["Hogwarts House", "First Name", "Last Name", "Birthday", "Index"])
    y_true = test_data["Hogwarts House"]

    # Define numerical and categorical columns
    numerical_cols = X_test.select_dtypes(
        include=["float64", "int64"]).columns.tolist()
    categorical_cols = X_test.select_dtypes(
        include=["object"]).columns.tolist()

    # Load preprocessor
    preprocessor = pkl.load(open('preprocessor.pkl', 'rb'))
    X_test_preprocessed = preprocessor.transform(X_test)

    # Load trained weights
    trained_weights = np.load('trained_weights.npy', allow_pickle=True).item()

    # Predict using trained models
    models = {house: LogisticRegressionSGD(weights) for house, weights in trained_weights.items()}
    y_pred = predict_multi_class_prob(models, X_test_preprocessed)

    # Save the predictions to houses.csv
    houses_output = pd.DataFrame({
        "Index": test_data["Index"],
        "Hogwarts House": y_pred
    })
    houses_output.to_csv('houses.csv', index=False)

print("Predictions saved to houses.csv.")
