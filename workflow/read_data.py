import pandas as pd
import re
from datetime import datetime


GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]


def read_users(fn):
    """
    Read the `users` file
    :param fn: filename
    :return: pandas dataframe
    """
    users = pd.read_table(fn, sep='::', engine='python', encoding='latin-1',
                          names=['UserID', 'Gender', 'Age', 'Occupation', 'ZipCode'])

    return users


def read_movies(fn):
    """
    Read the `movies` file
    :param fn: filename
    :return: pandas dataframe
    """
    movies = pd.read_table(fn, sep='::', engine='python', encoding='latin-1',
                           names=['MovieID', 'Title', 'Genres'])

    return movies


def read_ratings(fn):
    """
    Read the `ratings` file
    :param fn: filename
    :return: pandas dataframe
    """
    ratings = pd.read_table(fn, sep='::', engine='python', encoding='latin-1',
                            names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

    return ratings


def get_user_age():
    """
    Get the user age groups
    :return: pandas series
    """
    return pd.Series(["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"],
                     index=[1, 18, 25, 35, 45, 50, 56], name='AgeGroup')


def get_user_occupation():
    """
    Get the user occupation
    :return: pandas series
    """
    return pd.Series(["other or not specified", "academic/educator", "artist", "clerical/admin", "college/grad student",
                      "customer service", "doctor/health care", "executive/managerial", "farmer", "homemaker",
                      "K-12 student", "lawyer", "programmer", "retired", "sales/marketing", "scientist",
                      "self-employed", "technician/engineer", "tradesman/craftsman", "unemployed", "writer"],
                     index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                     name='OccupationDetailed')


def encode_movie_genres(movies_df, drop_genres_column=True):
    """
    Genres encoding with 0 and 1
    :param movies_df: pandas dataframe
    :param drop_genres_column: remove the original column after the encoding
    :return: pandas dataframe
    """
    res = movies_df.join(pd.DataFrame(0, index=movies_df.index, columns=[f"Genre_{x}" for x in GENRES]))

    for idx, r in res.iterrows():
        genres = r.Genres.split('|')
        for g in genres:
            res.loc[idx, f"Genre_{g}"] = 1

    if drop_genres_column:
        res.drop(columns=['Genres'], inplace=True)

    return res


def extract_movie_year(movies_df, remove_year_from_title=True):
    """
    Extract a movie year from the title
    :param movies_df: pandas dataframe
    :param remove_year_from_title: remove a year from the original Title column
    :return: pandas dataframe
    """
    pattern = re.compile(r"(.*)\S(\d{4})")
    movies_df['Year'] = movies_df.Title.map(lambda s: int(pattern.search(s).group(2)))

    if remove_year_from_title:
        movies_df.Title = movies_df.Title.map(lambda s: pattern.search(s).group(1))
    return movies_df


def get_rating_datetime(ratings_df, remove_timestamp_column=True):
    """
    Convert timestamp to datetime
    :param ratings_df: pandas dataframe
    :param remove_timestamp_column: remove the original Timestamp column
    :return: pandas dataframe
    """

    ratings_df['DateTime'] = ratings_df.Timestamp.map(datetime.fromtimestamp)

    if remove_timestamp_column:
        ratings_df.drop(columns=['Timestamp'], inplace=True)

    return ratings_df


def add_movie_descriptions(movies_df, descriptions_fn):
    """
    Load the descriptions from the file and add them to the dataframe
    :param movies_df: pandas dataframe
    :param descriptions_fn: filename
    :return: pandas dataframe
    """
    descriptions = pd.read_table(descriptions_fn, sep='::', engine='python', encoding='latin-1',
                                 names=['MovieID', 'Description'])

    movies_df = movies_df.join(descriptions, rsuffix="_other").drop(columns=['MovieID_other'])
    movies_df.Description.fillna("", inplace=True)

    return movies_df


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

