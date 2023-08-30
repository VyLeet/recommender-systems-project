
import argparse

from workflow.abstract_task import AbstractTask, DATA_DIR
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Evaluate(AbstractTask):
    def __init__(self, model=None):
        super().__init__(ids_file=DATA_DIR / 'test.ids')
        if model is not None:
            self.model = model

    @staticmethod
    def print_metrics(gt, predictions, model=None):
        print('---------------------------------------------')
        if model:
            print(f'Testing model: {repr(model)}')
        else:
            print('Testing model')

        print("\nMetrics:")
        print(f"MAE:  {mean_absolute_error(gt, predictions):.3f}")
        print(f"RMSE: {mean_squared_error(gt, predictions, squared=False):.3f}")
        print('---------------------------------------------')

    def run(self):
        predictions = self.model.predict(self.ratings.drop(columns='Rating'))
        self.print_metrics(gt=self.ratings.Rating, predictions=predictions, model=self.model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_file', required=True, type=str,
                        help='Serialized model')

    script_args = parser.parse_args()

    evaluate_task = Evaluate()
    evaluate_task.load_model(script_args.model_file)
    evaluate_task.run()
