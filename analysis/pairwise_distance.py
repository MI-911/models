import json
import os
from itertools import combinations, product

from networkx import shortest_path_length
from numpy import mean
from tqdm import tqdm

from analysis.movielens import ratings
from data.graph_loader import load_graph
from data.training import cold_start


def list_to_pairs(lst):
    return list(combinations(lst, r=2))


def handle_pairs(G, pairs):
    user_path_lengths = []
    for first, second in pairs:
        if first in G and second in G:
            user_path_lengths.append(shortest_path_length(G, first, second))

    return user_path_lengths


def analyse(G, out_file, user_pairs):
    path_lengths = []
    mean_path_lengths = []

    for pairs in tqdm(user_pairs):
        lengths = handle_pairs(G, pairs)

        path_lengths += lengths
        if lengths:
            mean_path_lengths.append(mean(lengths))

    with open(os.path.join('results', out_file), 'w') as fp:
        json.dump({
            'path_lengths': path_lengths,
            'mean_path_lengths': mean_path_lengths
        }, fp)


def analyse_mindreader(G):
    u_r_map, _, m_uri_map, _ = cold_start(
        from_path='../data/mindreader/user_ratings_map.json',
        conversion_map={
            -1: None,
            0: None,  # Ignore don't know ratings
            1: 1
        },
        split_ratio=[100, 0]
    )

    index_movie_map = {value: key for key, value in m_uri_map.items()}
    user_uris = []

    for user in u_r_map:
        liked = [index_movie_map[movie_id] for movie_id, rating in u_r_map[user]['movies'] if rating == 1]
        disliked = [index_movie_map[movie_id] for movie_id, rating in u_r_map[user]['movies'] if rating == -1]

        # uri_combinations = product(liked, disliked)
        uri_combinations = combinations(liked, r=2)

        user_uris.append(list(uri_combinations))

    analyse(G, 'mindreader_pairwise_distance.json', user_uris)


def analyse_movielens(G):
    combinations_list = list()
    user_ids = set(ratings['userId'])

    for user_id in user_ids:
        user_ratings = ratings[ratings.userId == user_id]

        user_likes = list(user_ratings[user_ratings.liked]['uri'])
        user_dislikes = list(user_ratings[~user_ratings.liked]['uri'])

        # combinations_list.append(list_to_pairs(user_likes))
        combinations_list.append(product(user_likes, user_dislikes))

    analyse(G, 'movielens_pairwise_distance.json', combinations_list)


if __name__ == '__main__':
    graph = load_graph(graph_path='../data/graph/triples.csv', directed=False, exclude_relations=['FROM_DECADE'])

    # analyse_mindreader(graph)
    analyse_movielens(graph)

