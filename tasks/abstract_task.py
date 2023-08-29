from abc import ABCMeta, abstractmethod
from tasks.read_data import read_ratings, read_users, read_movies, encode_movie_genres, extract_movie_year, \
    add_movie_descriptions, get_user_age, get_user_occupation, get_rating_datetime, subset_ratings
from pathlib import Path
import pickle


DATA_DIR = Path(__file__).parent.parent / 'data'


class AbstractTask(object):
    """
    The AbstractTask class loads data and creates interfaces for loading/saving a model. It is parent class for
    the Train, Evaluation and Inference pipelines
    """

    __metaclass__ = ABCMeta

    def __init__(self, ids_file):
        self.data_dir = DATA_DIR

        self.model = None

        self.users = self.load_users()
        self.movies = self.load_movies()
        self.ratings = self.load_ratings(ids_file)

    def load_movies(self, add_descriptions=True):
        movies = read_movies(self.data_dir / 'movies.dat')
        movies = encode_movie_genres(movies, drop_genres_column=True)
        movies = extract_movie_year(movies, remove_year_from_title=True)
        if add_descriptions:
            movies = add_movie_descriptions(movies, self.data_dir / 'movie_descriptions.dat')
        return movies

    def load_users(self):
        users = read_users(self.data_dir / 'users.dat')
        users = users.join(get_user_age(), on='Age')
        users = users.join(get_user_occupation(), on='Occupation')
        return users

    def load_ratings(self, ids_file, convert_datetime=False):
        ratings = read_ratings(self.data_dir / 'ratings.dat')
        if convert_datetime:
            ratings = get_rating_datetime(ratings, remove_timestamp_column=False)
        ratings = subset_ratings(self.data_dir / ids_file, ratings)
        return ratings

    def dump_model(self, fn):
        assert self.model is not None

        with open(fn, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, fn):
        with open(fn, 'rb') as f:
            self.model = pickle.load(f)

    @abstractmethod
    def run(self):
        raise NotImplementedError('The `run` method is not implemented')

