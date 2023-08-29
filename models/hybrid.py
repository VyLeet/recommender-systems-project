import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from models.abstract_model import AbstractModel
from models.collaborative import CollaborativeRecommender
from models.content_based import ContentBasedRecommender
from models.matrix_factorization import MatrixFactorizationRecommender


# Hybrid weighted ensemble recommender system
class HybridRecommender(AbstractModel):
    def __init__(self, users, movies):
        super().__init__(users, movies)

        # List the models taking part in ensemble
        self.models: list = [
            CollaborativeRecommender(users, movies),
            ContentBasedRecommender(users, movies),
            MatrixFactorizationRecommender(users, movies),
        ]

        # Initialize the weights equally
        self.weights = np.ones(len(self.models)) / len(self.models)

    def fit(self, X, y, should_learn_weights=False):
        # Fit the models
        [model.fit(X, y) for model in self.models]

        if should_learn_weights:

            def loss(weights):
                predictions = self.predict(X, weights)
                return mean_squared_error(y, predictions)

            result = minimize(loss, self.weights, method="BFGS")

            self.weights = result.x

    def predict(self, X, weights=None):
        if weights is None:
            weights = self.weights
        predictions = np.array([model.predict(X) for model in self.models]).T
        return predictions @ weights
