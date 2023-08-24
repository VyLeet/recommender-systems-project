from abc import ABCMeta, abstractmethod


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
