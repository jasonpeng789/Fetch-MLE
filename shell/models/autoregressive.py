import numpy as np

class AutoRegressiveModel:
    """
    A simple implementation of an AutoRegressive (AR) Model.

    This model is designed to perform linear regression tasks using the least squares method. It fits a linear model
    with coefficients to minimize the residual sum of squares between the observed targets in the dataset and the 
    targets predicted by the linear approximation.

    Attributes:
    -----------
    coefficients : numpy.ndarray
        The coefficients of the regression model. It is set to None initially and computed in the `fit` method.

    Methods:
    --------
    fit(X, y):
        Fits the AR model to the given data.

    predict(X):
        Predicts target values using the fitted AR model for given input data.
    """
    def __init__(self):
        """
        Initializes the AutoRegressiveModel instance with default attributes.
        """
        self.coefficients = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the model to the provided data using the Normal Equation.

        Parameters:
        X (numpy.ndarray): A 2D array where each row represents an observation and 
                           each column represents a variable. Should not include the bias term.
        y (numpy.ndarray): A 1D array of target values corresponding to each observation in X.

        Returns:
        None: Modifies the model in-place by setting the coefficients based on the input data.
        """
        # Add a bias term with a column of ones
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        # Compute the coefficients using the Normal Equation
        self.coefficients = np.linalg.pinv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for given input data using the fitted model.

        Parameters:
        X (numpy.ndarray): A 2D array where each row represents an observation and 
                           each column represents a variable. Should not include the bias term.

        Returns:
        numpy.ndarray: A 1D array of predicted values corresponding to each observation in X.
        """
        # Add a bias term for prediction
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        # Predict using the coefficients
        return X_bias.dot(self.coefficients)
    
