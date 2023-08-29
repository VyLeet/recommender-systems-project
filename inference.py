import argparse

from tasks.abstract_task import AbstractTask, DATA_DIR


class Inference(AbstractTask):
    def __init__(self, output_file, model=None):
        super().__init__(ids_file=DATA_DIR / 'inference.ids')
        self.output_file = output_file
        if model is not None:
            self.model = model

    def run(self):
        data = self.ratings.drop(columns='Rating')
        predictions = self.model.predict(data)

        output = data.copy()
        output['Rating'] = predictions
        output.to_csv(self.output_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_file', required=True, type=str,
                        help='Serialized model')
    parser.add_argument('-o', '--output_file', required=True, type=str,
                        help='Output CSV file')

    script_args = parser.parse_args()

    inference_task = Inference(script_args.output_file)
    inference_task.load_model(script_args.model_file)
    inference_task.run()



