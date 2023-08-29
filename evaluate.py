
from tasks.abstract_task import AbstractTask, DATA_DIR


class Evaluate(AbstractTask):
    def __init__(self, model=None):
        super().__init__(ids_file=DATA_DIR / 'test.ids')
        if model is not None:
            self.model = model


if __name__ == '__main__':
    pass
