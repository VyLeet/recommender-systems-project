from evaluation.read_data import read_ratings, read_users, read_movies, encode_movie_genres, extract_movie_year, \
    add_movie_descriptions, get_user_age, get_user_occupation, get_rating_datetime
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_DIR = Path(__file__).parent.parent / 'data'


class EvaluationFramework:

    def __init__(self):

        ratings = read_ratings(DATA_DIR / 'ratings.dat')
        ratings = get_rating_datetime(ratings, remove_timestamp_column=False)
        self.train_ratings = self.subset_ratings(DATA_DIR / 'train.ids', ratings)
        self.test_ratings = self.subset_ratings(DATA_DIR / 'test.ids', ratings)

        users = read_users(DATA_DIR / 'users.dat')
        users = users.join(get_user_age(), on='Age')
        users = users.join(get_user_occupation(), on='Occupation')
        self.users = users

        movies = read_movies(DATA_DIR / 'movies.dat')
        movies = encode_movie_genres(movies, drop_genres_column=True)
        movies = extract_movie_year(movies, remove_year_from_title=True)
        self.movies = add_movie_descriptions(movies, DATA_DIR / 'movie_descriptions.dat')

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

    @staticmethod
    def print_metrics(gt, predictions, model=None):
        print('---------------------------------------------')
        if model:
            print(f'Testing model: {type(model).__name__}')
        else:
            print('Testing model')

        print(f"MAE:  {mean_absolute_error(gt, predictions):.3f}")
        print(f"RMSE: {mean_squared_error(gt, predictions, squared=False):.3f}")
        print('---------------------------------------------')

    def evaluate(self, model_cls, model_params=None):
        model_params = model_params or {}
        model = model_cls(users=self.users, movies=self.movies, **model_params)

        model.fit(
            self.train_ratings.drop(columns='Rating'),
            self.train_ratings.Rating
        )
        predictions = model.predict(self.test_ratings.drop(columns='Rating'))
        self.print_metrics(gt=self.test_ratings.Rating, predictions=predictions, model=model)





