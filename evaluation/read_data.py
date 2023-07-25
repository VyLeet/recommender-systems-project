import pandas as pd


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
                     index=[1, 18, 25, 35, 45, 50, 56])


def get_user_occupation():
    """
    Get the user occupation
    :return: pandas series
    """
    return pd.Series(["other or not specified", "academic/educator", "artist", "clerical/admin", "college/grad student",
                      "customer service", "doctor/health care", "executive/managerial", "farmer", "homemaker",
                      "K-12 student", "lawyer", "programmer", "retired", "sales/marketing", "scientist",
                      "self-employed", "technician/engineer", "tradesman/craftsman", "unemployed", "writer"],
                     index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])


