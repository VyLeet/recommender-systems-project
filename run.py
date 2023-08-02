from evaluation.evaluation import EvaluationFramework
from models.matrix_factorization import MatrixFactorizationRecommender
from models.baseline import BaselineRecommender
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True,
                        choices=['base', 'content_based', 'collaborative', 'matrix_factorization'],
                        help='Model type')
    script_args = parser.parse_args()

    ef = EvaluationFramework()

    if script_args.model == 'base':
        model_class = BaselineRecommender
    elif script_args.model == 'matrix_factorization':
        model_class = MatrixFactorizationRecommender

    ef.evaluate(model_class)