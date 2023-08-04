import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from models.abstract_model import AbstractModel


class ContentBaseRecommender(AbstractModel):

    def __init__(self, users, movies, use_tfidf=True, tfidf_max_features=50):
        super().__init__(users, movies)

        self.use_tfidf = use_tfidf
        self.tfidf_max_features = tfidf_max_features

        self.user_features = self.get_user_features()
        self.movie_features = self.get_movie_features()

        self.user_combined_features = None
        self.movie_combined_features = None

    def get_tfidf_movie_features(self):
        tfidf = TfidfVectorizer(stop_words='english', max_features=50)
        tfidf_matrix = tfidf.fit_transform(self.movies.Title + " " + self.movies.Description)
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])
        return tfidf_df.set_index(self.movies.MovieID)

    def get_movie_features(self):
        features = self.movies.loc[:, self.movies.columns.str.startswith('Genre_')].set_index(self.movies.MovieID)
        if self.use_tfidf:
            features = features.join(self.get_tfidf_movie_features())

        return features

    def get_user_features(self):
        users_features = pd.get_dummies(self.users.set_index('UserID')[['Occupation', 'Gender', 'Age']],
                                        columns=['Occupation', 'Gender'], prefix=['occupation', 'gender'], dtype=int)

        return users_features

    @staticmethod
    def weight_features_by_ratings(src_features, scr_id_column, dst_id_column, ratings):
        dst_weights = ratings.groupby(dst_id_column)['Rating'].transform(lambda x: x / x.sum())
        dst_features = ratings.join(src_features, on=scr_id_column)[src_features.columns]
        dst_features = dst_features.mul(dst_weights, axis=0)
        return dst_features.groupby(ratings[dst_id_column]).sum()

    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def fit(self, X, y):
        X['Rating'] = y
        map_movie_features_to_users = self.weight_features_by_ratings(self.movie_features, 'MovieID', 'UserID', X)
        map_user_features_to_movies = self.weight_features_by_ratings(self.user_features, 'UserID', 'MovieID', X)

        self.user_combined_features = self.user_features.join(map_movie_features_to_users).fillna(0.0)
        self.movie_combined_features = map_user_features_to_movies.join(self.movie_features, how='right').fillna(0.0)

    def predict(self, X):
        predicted_ratings = np.zeros(X.shape[0])

        for i, (_, r) in enumerate(X.iterrows()):
            user_vector = self.user_combined_features.loc[r.UserID]
            movie_vector = self.movie_combined_features.loc[r.MovieID]

            predicted_ratings[i] = self.cosine_similarity(user_vector, movie_vector) * 4 + 1  # map 0-1 to ratings 1-5

        return predicted_ratings


