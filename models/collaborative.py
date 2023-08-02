import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from models.abstract_model import AbstractModel


class CollaborativeRecommender(AbstractModel):

    def __init__(self, users, movies):
        super().__init__(users, movies)
        self.matrix = None
        self.similarity_mtx = None

    def fit(self, X, y):

        # create a matrix for user similarity calculation
        # add user features (gender, age, occupation) for unseen users' similarity (since we have new users more
        # often than the new movies)
        users_features = pd.get_dummies(self.users.set_index('UserID')[['Occupation', 'Gender', 'Age']],
                                        columns=['Occupation', 'Gender'], prefix=['occupation', 'gender'])
        train_ratings = pd.pivot_table(X, index='MovieID', columns='UserID', values='Rating')




        X['Rating'] = y
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
