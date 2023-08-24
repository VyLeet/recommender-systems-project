import argparse
from pathlib import Path

from evaluation.read_data import subset_ratings, read_ratings, get_rating_datetime

DATA_DIR = Path(__file__).parent / 'data'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-mf', '--model_file', required=True, help='Model type', type=str)
    script_args = parser.parse_args()

    ratings = read_ratings(DATA_DIR / 'ratings.dat')
    ratings = get_rating_datetime(ratings, remove_timestamp_column=False)
    inference_ratings = subset_ratings(DATA_DIR / 'inference.ids', ratings)



