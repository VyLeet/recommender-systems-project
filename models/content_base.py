from abc import ABC
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from abstract_model import AbstractModel

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
        CV = CountVectorizer()
        similarity_matrix = CV.fit_transform(self.df[y])
        cosine_similarity = cosine_similarity(similarity_matrix)

        # Movies index mapping
        indices = pd.Series(self.df.index, index=self.df[y])

        # Get the index of the movie that matches the title
        movie_idx = indices[X]

        movie_scores = list(enumerate(cosine_similarity[movie_idx]))
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
