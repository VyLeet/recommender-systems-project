from abc import ABCMeta, abstractmethod


class AbstractModel(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError()
    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()
