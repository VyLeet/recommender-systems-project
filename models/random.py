from models.abstract_model import AbstractModel
import numpy as np


class RandomRecommender(AbstractModel):

    def __init__(self, users, movies, random_seed=42):
        super().__init__(users, movies)
        np.random.seed(random_seed)

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.random.randint(1, 5, X.shape[0])
