""" This module runs Flask"""

from flask import Flask, render_template, request
from recommender import user_array, get_movies_nmf, get_movies_cosim, get_movies_cosim_item, get_movies_cosim_item_mix
from sqlalchemy import create_engine
from config import USER, PASS

conn_string = f'postgres://{USER}:{PASS}@localhost:5432/movies'
PG = create_engine(conn_string)


app = Flask(__name__)


@app.route('/')
@app.route('/index')
def hello_world():
    return render_template('index.html')


@app.route('/recommender')
def recommender():
    user_input = dict(request.args)
    user_input_movies = list(user_input.values())[:-1]
    user_input_movies_fuzzed = []
    for movie in user_input_movies:
        movie_name_query = f"""SELECT title, "movieId",
                        ts_rank_cd(to_tsvector('english', movies.title), to_tsquery('''{movie}'':*'))
                        AS score
                        FROM movies
                        WHERE to_tsvector('english', movies.title) @@ to_tsquery('''{movie}'':*')
                        ORDER BY score DESC;"""
        result = PG.execute(movie_name_query).fetchall()[0][0]
        user_input_movies_fuzzed.append(result)

    user_input_ratings = [5] * len(user_input_movies_fuzzed)
    user_input = dict(zip(user_input_movies_fuzzed, user_input_ratings))

    if dict(request.args)['model'] == 'NMF':
        new_array = user_array(user_input, nmf=True)
        result_list = get_movies_nmf(new_array)

    if dict(request.args)['model'] == 'Cosim':
        new_array = user_array(user_input)
        result_list = get_movies_cosim(new_array)

    if dict(request.args)['model'] == 'Cosim Item':
        result_list = get_movies_cosim_item(user_input_movies_fuzzed)

    if dict(request.args)['model'] == 'Cosim Item Mix':
        result_list = get_movies_cosim_item_mix(user_input_movies_fuzzed)

    return render_template('recommender.html',
                            result_html=result_list,
                            user_input=user_input)
