from abc import ABC
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from abstract_model import AbstractModel


class ContentBaseRecommender(AbstractModel, ABC):
    def __init__(self, df):
        self.matrix = None


    def fit(self, X, y):
        # Compute similarity matrix and cosine similarity
        # Use TF/IDF model for content base recommendations
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(X['Genres'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        self.matrix = cosine_sim


    def predict(self, X):
        titles = X["Title"].values
        indices = pd.Series(titles.index, index=titles)

        predicted_ratings = np.zeros(len(X))

        for i in range(len(X)):
            idx = indices[i]
            sim_scores = list(enumerate(self.matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:10]
            movie_indices = [i[0] for i in sim_scores]
            predicted_ratings[i] = titles.iloc[movie_indices]

        return predicted_ratings
