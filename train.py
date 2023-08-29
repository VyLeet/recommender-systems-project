from tasks.model_selector import ModelSelector
from tasks.abstract_task import AbstractTask, DATA_DIR

class Train(AbstractTask):
    def __init__(self):
        super().__init__(ids_file=DATA_DIR / 'train.ids')

    def run(self):
        self.model.fit(
            self.ratings.drop(columns='Rating'),
            self.ratings.Rating
        )

if __name__ == '__main__':

    train_task = Train()
    train_task.run()