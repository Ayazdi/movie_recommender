"""
  This module read and clean the data into a dataframe foramt.
  Then, train and save the models.
 """
import pandas as pd
import pickle
from sklearn.decomposition import NMF
from scipy.spatial import distance
from sqlalchemy import create_engine
from config import POSTGRES
import re
import sys
from os.path import dirname
sys.path.append(dirname("./data/"))


PG = create_engine(POSTGRES)


def year(year):
    """Takes the year from the title"""
    return re.findall(r"\((\d{4})\)$", year)


def title(movie):
    """ Removes the year from the title"""
    return re.findall(r"(.+) \(", movie)


def read_and_transform():
    """
    Read and transform the csv files
    """

    movies = pd.read_csv('./data/movies.csv')
    # Make a new year column and remove the year form the title
    movies['year'] = movies['title'].apply(year)
    movies['title'] = movies['title'].apply(title)
    movies['title'] = movies['title'].apply(pd.Series)
    movies['year'] = movies['year'].apply(pd.Series)

    rating = pd.read_csv('./data/ratings.csv')
    rating['timestamp'] = pd.to_datetime(rating['timestamp'], unit='s')

    tags = pd.read_csv('./data/tags.csv')
    tags['timestamp'] = pd.to_datetime(tags['timestamp'], unit='s')

    links = pd.read_csv('./data/links.csv')

    # check if we need this
    movie_dict = pd.Series(movies.movieId.values, index=movies.title).to_dict()
    matrix = pd.pivot_table(rating, values='rating',
                            index='userId', columns='movieId')

    return movies, rating, tags, links, movie_dict, matrix


movies, rating, tags, links, movie_dict, matrix = read_and_transform()


def add_to_database(movies, rating, tags, links):
    """Add dataframes to Postgres database"""
    movies.to_sql('movies', PG)
    tags.to_sql('tags', PG)
    rating.to_sql('ratings', PG)
    links.to_sql('links', PG)


def train_and_save_nmf():
    """ Train negative matrix factorization model and save it as pickle file"""
    matrix = matrix.fillna(2.5)
    model = NMF(n_components=100, init='random',
                random_state=10, l1_ratio=0.01)
    model.fit(matrix)
    pickle.dump(model, open("nmf_model.sav", 'wb'))


def nmf_model():
    """
    Load NMF model, return the model and movie weights
    """

    model = pickle.load(open("nmf_model.sav", 'rb'))
    Q = model.components_  # movie-genre matrix (movie weights)

    return model, Q


def train_cosim_item_based_model():
    """
    Train a cosimilarity matrix based on items(movies) and save it as CSV file
    """
    matrix_i = pd.pivot_table(rating, values='rating',
                              index='movieId', columns='userId')
    # Only movies with at least 5 ratings
    matrix_i = matrix_i[matrix_i.isna().sum(axis=1) < 605]
    matrix_f = matrix_i.fillna(0)
    vector = distance.pdist(matrix_f, 'cosine')
    sim_mat = 1 - distance.squareform(vector)
    sim_mat = pd.DataFrame(sim_mat, index=matrix_i.index,
                           columns=matrix_i.index)
    sim_mat.to_csv('sim_matrix_movie_based.csv')


def read_cosim_item_based_model():
    """
    Load the cosimilarity matrix based on items form the CSV file into
    DataFrame and return it
    """
    sim_mat = pd.read_csv('sim_matrix_movie_based.csv', index_col=0)
    return sim_mat


if __name__ == '__main__':
    add_to_database(movies, rating, tags, links)
    read_and_transform()
    train_and_save_nmf()
    train_cosim_item_based_model()
