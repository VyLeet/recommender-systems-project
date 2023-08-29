from evaluation.evaluation import EvaluationFramework
import argparse
from tasks.model_selector import ModelSelector


if __name__ == '__main__':

    ms = ModelSelector()
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True,
                        choices=ms.get_keys_list(),
                        help='Model type', nargs="+")
    script_args = parser.parse_args()

    ef = EvaluationFramework()
    for m in script_args.model:
        ef.evaluate(ms[m])
