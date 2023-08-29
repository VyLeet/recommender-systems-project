
from tasks.abstract_task import AbstractTask, DATA_DIR


class Evaluate(AbstractTask):
    def __init__(self):
        super().__init__(ids_file=DATA_DIR / 'test.ids')


if __name__ == '__main__':
    pass
