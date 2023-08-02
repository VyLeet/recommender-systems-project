from evaluation.evaluation import EvaluationFramework
from models.matrix_factorization import MatrixFactorizationRecommender


if __name__ == '__main__':
    ef = EvaluationFramework()
    ef.evaluate(MatrixFactorizationRecommender)