
from evaluation.read_data import read_ratings
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'


class EvaluationFramework:

    def __init__(self):

        ratings = read_ratings(DATA_DIR / 'ratings.dat')
        self.train_ratings = self.subset_ratings(DATA_DIR / 'train.ids', ratings)
        self.test_ratings = self.subset_ratings(DATA_DIR / 'test.ids', ratings)

    @staticmethod
    def subset_ratings(ids_file, ratings):
        """
        The method reads a file with user and movie ids and filters the rating dataset
        :param ids_file: filename
        :param ratings: pandas dataframe with ratings
        :return: filtered pandas dataframe
        """
        with open(ids_file, 'r') as f:
            ids = {tuple(map(int, l.strip().split('\t'))) for l in f}

        return ratings.loc[ratings.apply(lambda r: (r.UserID, r.MovieID) in ids, axis=1)]

    def print_metrics(self, gt, predictions):
        pass
    def evaluate(self, model_cls, model_params=None):
        model_params = model_params or {}
        model = model_cls(**model_params)

        model.fit(
            self.train_ratings.drop(columns='Rating'),
            self.train_ratings.Rating
        )
        predictions = model.predict(self.test_ratings.drop(columns='Rating'))
        self.print_metrics(gt=self.test_ratings.Ratings, predictions=predictions)





