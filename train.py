from tasks.model_selector import ModelSelector
from tasks.abstract_task import AbstractTask, DATA_DIR
import argparse
from evaluate import Evaluate


class Train(AbstractTask):
    def __init__(self, model_cls, script_arguments):
        super().__init__(ids_file=DATA_DIR / 'train.ids')

        parameters = self.get_model_parameters(script_arguments, model_cls)
        self.model = model_cls(users=self.users, movies=self.movies, **parameters)

    def run(self):
        self.model.fit(
            self.ratings.drop(columns='Rating'),
            self.ratings.Rating
        )

    @staticmethod
    def get_model_parameters(script_arguments, model_cls):
        parameters = {}
        cli_key = model_cls.get_cli_key()

        for sn in dir(script_arguments):
            if sn.startswith(cli_key + '.'):
                argument = sn.removeprefix(cli_key + '.')
                parameters[argument] = getattr(script_arguments, sn)

        return parameters


if __name__ == '__main__':

    ms = ModelSelector()
    parser = argparse.ArgumentParser(parents=ms.get_arg_parsers())
    parser.add_argument('-m', '--model', required=True,
                        choices=ms.get_keys_list(),
                        help='Model type')
    parser.add_argument('--save', required=False,
                        type=str, help='Dump model to a file')
    parser.add_argument('--evaluate', required=False,
                        help='Evaluate model after training', action='store_true')

    script_args = parser.parse_args()

    train_task = Train(ms[script_args.model], script_args)
    train_task.run()

    if script_args.save:
        train_task.dump_model(script_args.save)

    if script_args.evaluate:
        evaluate_task = Evaluate(model=train_task.model)
        evaluate_task.run()
