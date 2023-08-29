import argparse

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds

from models.abstract_model import AbstractModel


class MatrixFactorizationRecommender(AbstractModel):
    def __init__(self, users, movies, num_factors=50):
        super().__init__(users, movies)
        self.num_factors = num_factors
        self.U = None
        self.VT = None
        self.sigma = None

    def fit(self, X, y):
        X['Rating'] = y
        # Create a user-item matrix from the ratings data
        user_item_matrix = pd.pivot_table(
            X, values="Rating", index="UserID", columns="MovieID", fill_value=0
        )

        # Transpose the user-item matrix to get the item-user matrix
        item_user_matrix = user_item_matrix.T
        item_user_matrix_sparse = csc_matrix(item_user_matrix).astype(float)

        # Apply Singular Value Decomposition to factorize the item-user matrix
        U, sigma, VT = svds(item_user_matrix_sparse, k=self.num_factors)

        # Update the U, VT, and sigma in the class instance
        self.U = U
        self.VT = VT
        self.sigma = sigma

    def predict(self, X):
        # Get the user and movie IDs from the input data
        user_ids = X["UserID"].values
        movie_ids = X["MovieID"].values

        # Initialize an empty array to store the predicted ratings
        predicted_ratings = np.zeros(len(X))

        # Loop through each user-item pair in the input data
        for i in range(len(X)):
            user_id = user_ids[i]
            movie_id = movie_ids[i]

            # If the user or movie ID is not present in the training data, set the predicted rating to a default value (e.g., mean rating)
            if user_id not in self.users or movie_id not in self.movies:
                predicted_ratings[i] = 3.0
                continue

            # Get the index of the user and movie in the matrix factorization matrices
            user_idx = self.users.index(user_id)
            movie_idx = self.movies.index(movie_id)

            # Compute the predicted rating using the dot product of the corresponding rows in U and VT
            predicted_ratings[i] = np.dot(self.U[user_idx, :], self.VT[:, movie_idx])

        return predicted_ratings

    def __repr__(self):
        repr_str = super().__repr__()
        repr_str += f"\nNum factors: {self.num_factors}"

        return repr_str

    @classmethod
    def get_argument_parser(cls):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(f'--{cls.get_cli_key()}.num_factors', type=int, default=50)
        return parser
