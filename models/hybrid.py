import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from models.abstract_model import AbstractModel
from models.collaborative import CollaborativeRecommender
from models.content_base import ContentBaseRecommender
from models.matrix_factorization import MatrixFactorizationRecommender


class HybridRecommender(AbstractModel):
    def __init__(self, users, movies):
        super().__init__(users, movies)

        self.models: list = [
            CollaborativeRecommender(users, movies),
            ContentBaseRecommender(users, movies),
            MatrixFactorizationRecommender(users, movies),
        ]

        self.weights = np.ones(len(self.models)) / len(self.models)

    def fit(self, X, y):
        [model.fit(X, y) for model in self.models]

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models]).T
        return predictions @ np.array(self.weights)
