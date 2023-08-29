from abc import ABCMeta, abstractmethod
import re


def camel_to_snake(name):
    # https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


class AbstractModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, users, movies):
        self.users = users
        self.movies = movies

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()

    def __repr__(self):
        return type(self).__name__

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()

    @classmethod
    def get_cli_key(cls):
        return camel_to_snake(cls.__name__.removesuffix("Recommender"))
