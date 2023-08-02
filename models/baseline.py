from abstract_model import AbstractModel
import numpy as np


class BaselineRecommender(AbstractModel):

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.random.randint(1, 5, X.shape[0])
