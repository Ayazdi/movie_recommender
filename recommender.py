"""The module that recommend the movies based on each model"""

from read_and_train import nmf_model, read_and_transform, read_cosim_item_based_model
import pandas as pd
import numpy as np
from scipy.spatial import distance

movies, rating, tags, links, movie_dict, matrix = read_and_transform()


def user_array(movie_ratings, nmf=False):
    """
    Make a new user array with rating of the given movies.

    Movie_ratings (dict): name of the movie that the user want to
    rate as keys (str) and raitng of the user for the movie as
    values - the raings are always 5  (float)

    Returns series with rating of the movies
    """
    new_user = matrix.iloc[0, :]  # taking the first user as a new user
    if nmf:
        new_user[:] = 2.5
    else:
        new_user[:] = np.nan

    for k, v in movie_ratings.items():
        movie_id = list(movies['movieId'][movies['title'] == k])[0]
        new_user[new_user.index == movie_id] = v
    # new_user = new_user.fillna(matrix.mean())
    return new_user


def id_to_name(movie_id):
    """
    Convert a list of movie ids to a list of movie titles
    """
    movies_list = []
    for i in set(movie_id):

        name = list(movies['title'][movies['movieId'] == i])[0]
        movies_list.append(name)
    return movies_list


def name_to_id(movie_name):
    """
    Convert a list of movie titles to a list of movie id
    """
    watched_list_id = []
    for i in movie_name:
        id_ = list(movies['movieId'][movies['title'] == i])[0]
        watched_list_id.append(id_)
    return watched_list_id


def get_movies_nmf(new_user):
    """
    Recommend movies to a new user using NMF model
    retunrs a list with the names of 5 movies
    """
    # new_user[:] = 2.5
    model, Q = nmf_model()
    new_rating = model.transform([new_user])
    new_rating = np.dot(new_rating, Q)
    new_rating = new_rating.reshape(9724,)

    user_rating = pd.DataFrame()
    user_rating['initial'] = new_user
    user_rating['new_rating'] = new_rating

    user_rating = user_rating[user_rating['initial'] == 2.5]
    top_movie_id = user_rating.sort_values(
        'new_rating', ascending=False).head(5).index

    return id_to_name(top_movie_id)


def get_movies_cosim(new_user):
    """
    Recommend movies to a new user using Cosim model
    retunrs a list with the names of 5 movies
    """
    matrix_n = matrix[matrix.isna().sum(axis=1) < 9690]
    matrix_n = matrix_n.apply(lambda x: x - x.mean(), axis=1)
    matrix_n.loc['new', :] = new_user
    matrix_f = matrix_n.fillna(0)

    vector = 1 - distance.pdist(matrix_f, 'cosine')
    sim_mat = distance.squareform(vector)
    sim_mat = pd.DataFrame(sim_mat, index=matrix_n.index,
                           columns=matrix_n.index)

    # number of similar users to be consider
    sim_users = sim_mat.loc['new'].sort_values(ascending=False)[1:4].index
    sim_users = matrix.loc[sim_users]

    high_rate = list(sim_users.mean()[
                     (sim_users.mean().sort_values(ascending=False) == 5)].index)
    top_movie_id = sim_users[high_rate].isna().sum(
    ).sort_values(ascending=False).head(5).index

    return id_to_name(top_movie_id)


sim_mat_item = read_cosim_item_based_model()


def get_movies_cosim_item(watched_list_name):
    """
    Recommend movies to a new user using "Cosim Item Based" model.
    Retunrs a list of  movies that are similar to every individual
    movie in the watched_list
    """

    watched_list_id = name_to_id(watched_list_name)

    movie_id = []
    for i in watched_list_id:
        rec_movies = list(sim_mat_item.loc[i].sort_values(
            ascending=False)[1:6].index)
        for j in rec_movies:
            movie_id.append(int(j))

    for id_ in movie_id:
        if id_ in watched_list_id:
            movie_id.remove(id_)

    return id_to_name(movie_id)


def get_movies_cosim_item_mix(watched_list_name):
    """
    Recommend movies to a new user using "Cosim Item Based" model.
    Retunrs a list of  movies that are similar to the combination of
    all movies in the watched_list.
    """
    watched_list_id = name_to_id(watched_list_name)

    new_movie = sim_mat_item.loc[watched_list_id]
    new_movie.loc['new_movie'] = np.nan
    new_movie = new_movie.fillna(new_movie.mean())

    similar_movies_id = new_movie.loc['new_movie'].sort_values(
        ascending=False).head(8).index
    similar_movies_id = [int(x) for x in similar_movies_id]

    for id_ in watched_list_id:
        if id_ in similar_movies_id:
            similar_movies_id.remove(id_)

    return id_to_name(similar_movies_id)
