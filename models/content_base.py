from abc import ABC
import pandas as pd
from models.abstract_model import AbstractModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentBaseRecommender(AbstractModel, ABC):
    def __init__(self, df):
        self.indices = None
        self.df = df

    "" \
    " Fit method" \
    " X: none" \
    " y: column name" \
    ""
    def fit(self, X, y):
        # Compute similarity matrix and cosine similarity
        # Use TF/IDF model for content base recommendations
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.df[y].fillna(''))

        # Compute similarity matrix
        similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Movies index mapping
        indices = pd.Series(self.df.index, index=self.df[y])

        # Get the index of the movie that matches the title
        movie_idx = indices[X]

        movie_scores = list(enumerate(similarity_matrix[movie_idx]))
        movie_scores = sorted(movie_scores, key=lambda x: x[1], reverse=True)
        # start from 1, where 0 element is itself.
        # Top 5.
        movie_scores = movie_scores[1:5]

        # Set the movie indices
        self.indices = [i[0] for i in movie_scores]


    " Predict method " \
    " X: input value (e.g.: Title)"
    def predict(self, X):
        return self.df[X].iloc[self.indices]
