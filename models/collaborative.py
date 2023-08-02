import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from abstract_model import AbstractModel


class CollaborativeRecommender(AbstractModel):

    def __init__(self, df):
        self.matrix = None
        self.similarity_mtx = None
        self.df = df

    def fit(self, X, y):
        matrix = pd.pivot_table(X, index='MovieID', columns='UserID', values='Rating')
        matrix = matrix.fillna(0)

        distance_mtx = squareform(pdist(matrix, 'cosine'))
        similarity_mtx = 1 - distance_mtx

        self.matrix = matrix
        self.similarity_mtx = similarity_mtx


    def predict(self, X):
        user_ids = X['UserID'].values
        movie_ids = X['MovieID'].values

        predicted_ratings = np.zeros(len(X))

        for i in range(len(X)):
            user_rating = self.matrix.iloc[:, user_ids[i] - 1]
            movie_similarity = self.similarity_mtx[movie_ids[i] - 1]

            numerator = np.dot(user_rating, movie_similarity)
            denomination = movie_similarity[user_rating > 0].sum()
            predicted_ratings[i] = numerator / denomination

        return predicted_ratings
