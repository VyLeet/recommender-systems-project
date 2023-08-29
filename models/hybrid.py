import argparse

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from models.abstract_model import AbstractModel
from models.collaborative import CollaborativeRecommender
from models.content_based import ContentBasedRecommender
from models.matrix_factorization import MatrixFactorizationRecommender


# Hybrid weighted ensemble recommender system
class HybridRecommender(AbstractModel):
    def __init__(self, users, movies, learn_weights=False, collaborative_top_k=30, content_use_tfidf=True,
                 content_tfidf_max_features=50, matrix_factorization_num_factors=50):
        super().__init__(users, movies)

        # List the models taking part in ensemble
        self.models: list = [
            CollaborativeRecommender(users, movies, top_k=collaborative_top_k),
            ContentBasedRecommender(users, movies, use_tfidf=content_use_tfidf,
                                    tfidf_max_features=content_tfidf_max_features),
            MatrixFactorizationRecommender(users, movies, num_factors=matrix_factorization_num_factors),
        ]

        self.learn_weights = learn_weights
        # Initialize the weights equally
        self.weights = np.ones(len(self.models)) / len(self.models)

    def fit(self, X, y):
        # Fit the models
        [model.fit(X, y) for model in self.models]

        if self.learn_weights:

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

    @classmethod
    def get_argument_parser(cls):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(f'--{cls.get_cli_key()}.learn_weights', type=bool, default=False)
        parser.add_argument(f'--{cls.get_cli_key()}.collaborative_top_k', type=int, default=30)
        parser.add_argument(f'--{cls.get_cli_key()}.content_tfidf_max_features', type=int, default=50)
        parser.add_argument(f'--{cls.get_cli_key()}.content_use_tfidf', type=bool, default=True)
        parser.add_argument(f'--{cls.get_cli_key()}.matrix_factorization_num_factors', type=int, default=50)

        return parser

