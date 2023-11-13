import numpy as np

class AutoRegressiveModel:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        # Add a bias term with a column of ones
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        # Compute the coefficients using the Normal Equation
        self.coefficients = np.linalg.pinv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y)

    def predict(self, X):
        # Add a bias term for prediction
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        # Predict using the coefficients
        return X_bias.dot(self.coefficients)
    
