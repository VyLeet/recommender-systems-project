from tasks.model_selector import ModelSelector
from tasks.abstract_task import AbstractTask, DATA_DIR
import argparse
from evaluate import Evaluate


class Train(AbstractTask):
    def __init__(self, model_cls):
        super().__init__(ids_file=DATA_DIR / 'train.ids')
        self.model = model_cls(users=self.users, movies=self.movies)

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
    parser.add_argument('--save', required=False,
                        type=str, help='Dump model to a file')
    parser.add_argument('--evaluate', required=False,
                        type=bool, help='Evaluate model after training', store_true=True)

    script_args = parser.parse_args()

    train_task = Train(ms[script_args.model])
    train_task.run()

    if script_args.save:
        train_task.dump_model(script_args.save)

    if script_args.evaluate:
        evaluate_task = Evaluate()
        evaluate_task.run()
