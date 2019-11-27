import json
import os
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from itertools import combinations

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
        user_uris.append(list_to_pairs([index_movie_map[movie_id] for movie_id, rating in u_r_map[user]['movies']]))

    analyse(G, 'mindreader_pairwise_distance.json', user_uris)


def analyse_movielens(G):
    user_uris = ratings.where(ratings.rating > 3.5).groupby('userId')['uri'].apply(list).reset_index(name='uris')['uris']

    analyse(G, 'movielens_pairwise_distance.json', user_uris)


if __name__ == '__main__':
    graph = load_graph(graph_path='../data/graph/triples.csv', directed=False)

    analyse_mindreader(graph)
    # analyse_movielens(graph)

