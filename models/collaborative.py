import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from models.abstract_model import AbstractModel


class CollaborativeRecommender(AbstractModel):

    def __init__(self, users, movies, top_k=30):
        super().__init__(users, movies)
        self.matrix = None
        self.similarity_mtx = None
        self.top_k = top_k

    def fit(self, X, y):

        # create a matrix for user similarity calculation
        # add user features (gender, age, occupation) for unseen users' similarity (since we have new users more
        # often than the new movies)
        users_features = pd.get_dummies(self.users.set_index('UserID')[['Occupation', 'Gender', 'Age']],
                                        columns=['Occupation', 'Gender'], prefix=['occupation', 'gender'], dtype=int)
        X['Rating'] = y
        train_ratings = pd.pivot_table(X, index='UserID', columns='MovieID', values="Rating")
        self.matrix = users_features.join(train_ratings).fillna(0)

        distance_mtx = squareform(pdist(self.matrix, 'cosine'))
        self.similarity_mtx = 1 - distance_mtx
        np.fill_diagonal(self.similarity_mtx, 0.0)

    def predict(self, X):

        predicted_ratings = np.zeros(X.shape[0])

        for i, (_, r) in enumerate(X.iterrows()):
            user_similarity = self.similarity_mtx[:, self.matrix.index.get_loc(r.UserID)]
            neighborhood_idx = np.argsort(user_similarity)[-self.top_k:]

            if r.MovieID in self.matrix.columns:  # previously seen movie
                movie_train_ratings = self.matrix[r.MovieID].to_numpy()
                neighborhood_ratings = movie_train_ratings[neighborhood_idx]
            else:  # previously unseen movie
                neighborhood_ratings = 0

            if np.sum(neighborhood_ratings) > 0:
                neighborhood_similarity = user_similarity[neighborhood_idx]
                numerator = np.dot(neighborhood_ratings, neighborhood_similarity)
                denominator = neighborhood_similarity[neighborhood_ratings > 0].sum()

                predicted_ratings[i] = numerator / denominator
            else:
                neighborhood_all_ratings = \
                    self.matrix.iloc[neighborhood_idx, [isinstance(c, int) for c in self.matrix.columns]].to_numpy()

                if neighborhood_all_ratings.sum() > 0:
                    predicted_ratings[i] = neighborhood_all_ratings.sum() / (neighborhood_all_ratings > 0).sum()
                else:
                    predicted_ratings[i] = 3

        return predicted_ratings
