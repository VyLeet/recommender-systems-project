from tasks.model_selector import ModelSelector
from tasks.abstract_task import AbstractTask, DATA_DIR
import argparse


class Train(AbstractTask):
    def __init__(self, model_cls):
        super().__init__(ids_file=DATA_DIR / 'train.ids')
        self.model_cls = model_cls

    def run(self):
        self.model.fit(
            self.ratings.drop(columns='Rating'),
            self.ratings.Rating
        )


if __name__ == '__main__':

    ms = ModelSelector()
    parser = argparse.ArgumentParser(parents=ms.get_arg_parsers())
    parser.add_argument('-m', '--model', required=True,
                        choices=ms.get_keys_list(),
                        help='Model type')
    script_args = parser.parse_args()

    train_task = Train(ms[script_args.model])
    train_task.run()
