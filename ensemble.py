import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler


class EnsembleModel(BaseEstimator, ClassifierMixin):
    def __init__(self, models, weights=None):
        """
        Initialize the ensemble model.

        Args:
            models (list): List of trained models
            weights (list, optional): Weights for each model's prediction.
                                     Defaults to equal weights if not provided.
        """
        self.models = models
        self.weights = (
            weights if weights is not None else [1 / len(models)] * len(models)
        )

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Args:
            X (array-like): Input samples

        Returns:
            numpy.ndarray: Predicted class probabilities
        """
        # Get probabilities from each model
        model_probas = [model.predict_proba(X) for model in self.models]

        # Weighted average of probabilities
        weighted_probas = np.zeros_like(model_probas[0])
        for probas, weight in zip(model_probas, self.weights):
            weighted_probas += probas * weight

        # Normalize probabilities to ensure they sum to 1
        weighted_probas /= np.sum(weighted_probas, axis=1, keepdims=True)

        return weighted_probas

    def predict(self, X):
        """
        Predict class labels for X.

        Args:
            X (array-like): Input samples

        Returns:
            numpy.ndarray: Predicted class labels
        """
        # Use the probabilities to make predictions
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
