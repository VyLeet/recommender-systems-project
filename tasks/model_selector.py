
from models.abstract_model import AbstractModel
import models as m
import inspect
import importlib
import pkgutil


class ModelSelector:

    def __init__(self):
        self.models = {}

        for a in pkgutil.iter_modules([m.__name__]):

            module = importlib.import_module(name=f'{m.__name__}.{a.name}')

            for class_name, class_obj in inspect.getmembers(module, inspect.isclass):

                if issubclass(class_obj, AbstractModel) and class_obj != AbstractModel:
                    self.models[class_name] = class_obj



