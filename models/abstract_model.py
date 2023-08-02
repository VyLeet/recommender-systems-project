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
