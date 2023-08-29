
from models.abstract_model import AbstractModel
import models as m
import inspect
import importlib
import pkgutil
import os


class ModelSelector:

    def __init__(self):
        self.models = {}

        for submodule in pkgutil.iter_modules([os.path.dirname(m.__file__)]):

            module = importlib.import_module(name=f'{m.__name__}.{submodule.name}')

            for _, cls in inspect.getmembers(module, inspect.isclass):

                if issubclass(cls, AbstractModel) and cls != AbstractModel:
                    self.models[cls.get_cli_key()] = cls

    def get_keys_list(self):
        return list(self.models.keys())

    def __getitem__(self, key):
        if key not in self.models:
            raise NotImplementedError(f"Model for the `{key}` key is not implemented")

        return self.models[key]

