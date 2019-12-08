import json
from collections import defaultdict
from random import sample, shuffle, choice
import itertools

from sklearn.neighbors import NearestNeighbors

from data.training import cold_start
from scipy import spatial

import numpy as np

from utilities.metrics import ndcg_at_k, average_precision, hitrate
from utilities.util import filter_min_k, get_top_movies, get_entity_occurrence, prune_low_occurrence, split_users


def predict_movies(idx_movie, u_r_map, neighbour_weights, top_movies=None, popularity_bias=1, exclude=None):
    movie_weight = defaultdict(int)

    # Add popularity bias
    if top_movies:
        for movie in top_movies:
            movie_weight[movie] += popularity_bias

    for neighbour, weight in neighbour_weights:
        if not weight:
            continue

        for movie, rating in u_r_map[neighbour]['movies'] + u_r_map[neighbour]['test']:
            movie_uri = idx_movie[movie]
            movie_weight[movie_uri] += rating

    # Get weighted prediction and exclude excluded URIs
    predictions = sorted(list(movie_weight.items()), key=lambda x: x[1], reverse=True)
    return [head for head, rating in predictions if not exclude or head not in exclude]


def user_to_vector(ratings, entity_idx, idx_movie, include_test=False):
    vector = np.zeros(len(entity_idx))
    movie_ratings = [(entity_idx[idx_movie[head]], rating) for head, rating in ratings['movies'] + (ratings['test'] if include_test else [])]

    for idx, rating in ratings['entities'] + movie_ratings:
        vector[idx] = rating

    return vector


def run():
    dislike = 1
    unknown = None
    like = 5

    u_r_map, n_users, movie_idx, entity_idx = cold_start(
        from_path='../data/mindreader/user_ratings_map.json',
        conversion_map={
            -1: dislike,
            0: unknown,
            1: like
        },
        restrict_entities=None,
        split_ratio=[75, 25]
    )

    # Load entities
    entities = dict()
    for entity, name, labels in json.load(open('../data/mindreader/entities_clean.json')):
        entities[entity] = name

    # Add movies to the entity_idx map
    # Just makes it easier when constructing user vectors
    entity_count = len(entity_idx)
    for movie, idx in movie_idx.items():
        if movie not in entity_idx:
            entity_idx[movie] = entity_count
            entity_count += 1

    idx_entity = {value: key for key, value in entity_idx.items()}
    idx_movie = {value: key for key, value in movie_idx.items()}

    # Filter entities with only one occurrence
    entity_occurrence = get_entity_occurrence(u_r_map, idx_entity, idx_movie)
    u_r_map = prune_low_occurrence(u_r_map, idx_entity, idx_movie, entity_occurrence)

    # Split into train and test users
    all_users = list(u_r_map.items())
    train_users, test_users = split_users(all_users, train_ratio=0.75)

    # Create indices from train users
    user_idx = {}
    user_count = 0
    for user, _ in train_users:
        user_idx[user] = user_count
        user_count += 1

    # Populate user vectors
    user_vectors = {}
    for user, ratings in train_users:
        user_vectors[user] = user_to_vector(ratings, entity_idx, idx_movie, include_test=True)

    # For index lookup
    keys = list(user_vectors.keys())

    # kNN model
    model = NearestNeighbors()
    model.fit(list(user_vectors.values()))

    # Predict
    smoke_vector = np.zeros(len(entity_idx))
    smoke_vector[entity_idx['http://www.wikidata.org/entity/Q3772']] = like

    distances, indices = model.kneighbors(smoke_vector.reshape(1, len(smoke_vector)), 5)

    neighbours = list()
    for index in indices.squeeze():
        neighbours.append((keys[index], 1))

    predictions = predict_movies(idx_movie, u_r_map, neighbours)[:5]
    for uri in predictions:
        print(entities[uri])



if __name__ == '__main__':
    run()
