import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
#from abc import ABCMeta, abstractmethod
from models.abstract_model import AbstractModel
#import evaluation


class MatrixFactorizationRecommender(AbstractModel):
    def __init__(self, users, movies, num_factors=50):
        super().__init__(users, movies)
        self.num_factors = num_factors
        self.U = None
        self.VT = None
        self.sigma = None  # Maksym

    def fit(self, X, y):
        # Create a user-item matrix from the ratings data
        user_item_matrix = pd.pivot_table(
            X, values="Rating", index="UserID", columns="MovieID", fill_value=0
        )

        # Transpose the user-item matrix to get the item-user matrix
        item_user_matrix = user_item_matrix.T

        # Apply Singular Value Decomposition (SVD) to factorize the item-user matrix
        U, sigma, VT = svds(item_user_matrix, k=self.num_factors)

        # Convert sigma to a diagonal matrix
        sigma_diag = np.diag(sigma)

        # Update the U and VT matrices in the class instance
        self.U = U
        self.VT = VT
        self.sigma = sigma  # Maksym

    def predict(self, X):
        # Get user and item indices from the input data
        user_indices = X["UserID"] - 1  # Adjust to 0-based index
        item_indices = X["MovieID"] - 1  # Adjust to 0-based index

        # Estimate the missing ratings by multiplying U, sigma_diag, and VT
        predicted_ratings = np.dot(
            self.U[user_indices, :], np.dot(np.diag(self.sigma), self.VT[:, item_indices])
        )

        return predicted_ratings


def preprocess_data(users_df, movies_df, ratings_df):
    # Encode movie genres as binary columns
    movies_df_encoded = encode_movie_genres(movies_df)

    # Extract movie years from the title
    movies_df_processed = extract_movie_year(movies_df_encoded)

    return users_df, movies_df_processed, ratings_df


if __name__ == '__main__':

    # Your code for the MatrixFactorizationRecommender class
    # ...

    # Read the datasets
    users = read_users("users.dat")
    movies = read_movies("movies.dat")
    ratings = read_ratings("ratings.dat")

    # Preprocess the data
    users, movies, ratings = preprocess_data(users, movies, ratings)

    # Create the recommender model instance
    num_factors = 50
    model = MatrixFactorizationRecommender(num_factors=num_factors)

    # Call the fit method to train the model
    model.fit(movies, ratings)

    user_id = 1
    user_ratings = ratings[ratings["userId"] == user_id]
    user_unrated_movies = movies[~movies["movieId"].isin(user_ratings["movieId"])]
    user_unrated_movies["predicted_rating"] = model.predict(user_unrated_movies)
    recommended_movies = user_unrated_movies.sort_values(
        by="predicted_rating", ascending=False
    ).head(10)
    print(recommended_movies[["movieId", "title", "predicted_rating"]])

    ####################################################################


    # Example usage:
    # Assuming you have loaded the 'movies', 'ratings', and 'users' datasets into pandas DataFrames.
    # model = MatrixFactorizationRecommender(num_factors=50)
    # model.fit(ratings, users)
    # user_id = 1
    # user_ratings = ratings[ratings['userId'] == user_id]
    # user_unrated_movies = movies[~movies['movieId'].isin(user_ratings['movieId'])]
    # user_unrated_movies['predicted_rating'] = model.predict(user_unrated_movies)
    # recommended_movies = user_unrated_movies.sort_values(by='predicted_rating', ascending=False).head(10)
    # print(recommended_movies[['movieId', 'title', 'predicted_rating']])
