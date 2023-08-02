import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
#from abc import ABCMeta, abstractmethod
from models.abstract_model import AbstractModel
#import evaluation


class MatrixFactorizationRecommender(AbstractModel):
    def __init__(self, users, movies, num_factors=50):
        super().__init__(users, movies)
        self.num_factors = num_factors
        self.U = None
        self.VT = None
        self.sigma = None

    def fit(self, X, y):
        # Create a user-item matrix from the ratings data
        user_item_matrix = pd.pivot_table(
            X, values="Rating", index="UserID", columns="MovieID", fill_value=0
        )

        # Transpose the user-item matrix to get the item-user matrix
        item_user_matrix = user_item_matrix.T
        item_user_matrix_sparse = csc_matrix(item_user_matrix).astype(float)

        # Apply Singular Value Decomposition (SVD) to factorize the item-user matrix
        U, sigma, VT = svds(item_user_matrix_sparse, k=self.num_factors)

        # Update the U and VT matrices in the class instance
        self.U = U
        self.VT = VT
        self.sigma = sigma

    def predict(self, X):
        # Get user and item indices from the input data
        user_indices = X["UserID"] - 1  # Adjust to 0-based index
        item_indices = X["MovieID"] - 1  # Adjust to 0-based index

        # Filter out-of-bound indices
        valid_user_indices = user_indices[user_indices < self.U.shape[0]]
        valid_item_indices = item_indices[item_indices < self.VT.shape[1]]

        # Estimate the missing ratings only for valid indices
        predicted_ratings = np.zeros(len(X))
        return predicted_ratings
        predicted_ratings_mask = np.logical_and(
            user_indices < self.U.shape[0], item_indices < self.VT.shape[1]
        )

        # Compute the predicted ratings for valid indices
        predicted_ratings[predicted_ratings_mask] = np.dot(
            self.U[valid_user_indices, :],
            np.dot(np.diag(self.sigma), self.VT[:, valid_item_indices]),
        )

        return predicted_ratings
