from evaluation.evaluation import EvaluationFramework
from models.matrix_factorization import MatrixFactorizationRecommender
from models.content_base import ContentBaseRecommender
from models.random import RandomRecommender
from models.collaborative import CollaborativeRecommender
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True,
                        choices=['random', 'content_based', 'collaborative', 'matrix_factorization'],
                        help='Model type', nargs="+")
    script_args = parser.parse_args()

    ef = EvaluationFramework()

    for model_name in script_args.model:
        if model_name == 'random':
            model_class = RandomRecommender
        elif model_name == 'content_based':
            model_class = ContentBaseRecommender
        elif model_name == 'collaborative':
            model_class = CollaborativeRecommender
        elif model_name == 'matrix_factorization':
            model_class = MatrixFactorizationRecommender
        else:
            raise NotImplementedError()

        ef.evaluate(model_class)
