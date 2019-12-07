import json
from collections import defaultdict
from random import sample, shuffle, choice
import itertools
from data.training import cold_start
from scipy import spatial
from sklearn.neighbors import NearestNeighbors
import numpy as np

from utilities.metrics import ndcg_at_k, average_precision, hitrate
from utilities.util import filter_min_k, get_top_movies, get_entity_occurrence, prune_low_occurrence, split_users


def similarity(a, b):
    return np.linalg.norm(a) - np.linalg.norm(b)


def knn(user_vectors, user, own_vector, neighbours):
    similarities = []

    for other, other_vector in user_vectors.items():
        if other == user:
            continue

        similarities.append((other, similarity(own_vector, other_vector)))

    # Shuffle s.t. any secondary ordering is random
    shuffle(similarities)

    return sorted(similarities, key=lambda x: x[1], reverse=True)[:neighbours]


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
            movie_weight[movie_uri] += rating * weight

    # Get weighted prediction and exclude excluded URIs
    predictions = sorted(list(movie_weight.items()), key=lambda x: x[1], reverse=True)
    return [head for head, rating in predictions if not exclude or head not in exclude]


def get_variances(item_vectors):
    item_variances = {}

    for item, vector in item_vectors.items():
        item_variances[item] = np.var(vector[np.nonzero(vector)])

    return item_variances


def run():
    dislike = -1
    unknown = None
    like = 1

    u_r_map, n_users, movie_idx, entity_idx = cold_start(
        from_path='../data/mindreader/user_ratings_map.json',
        conversion_map={
            -1: dislike,
            0: unknown,
            1: like
        },
        restrict_entities=None,
        split_ratio=[100, 0]
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
    train_users, test_users = split_users(list(u_r_map.items()), train_ratio=0.75)

    # Create indices from train users
    user_idx = {}
    user_count = 0
    for user, _ in train_users:
        user_idx[user] = user_count
        user_count += 1

    # Initialize item vectors
    entity_vectors = {}
    for entity, idx in entity_idx.items():
        entity_vectors[entity] = np.zeros(len(user_idx))

    # Populate item vectors
    for user, ratings in train_users:
        idx = user_idx[user]

        entity_ratings = [(idx_entity[head], rating) for head, rating in ratings['entities']]
        movie_ratings = [(idx_movie[head], rating) for head, rating in ratings['movies']]
        uri_ratings = entity_ratings + movie_ratings

        for uri, rating in uri_ratings:
            entity_vectors[uri][idx] = rating

    # Print entities with highest variance, just for fun
    # item_variances = sorted(entity_vectors.items(), key=lambda item: item[1], reverse=True)

    # Get movie vectors
    movie_vectors = dict()
    for entity, vector in entity_vectors.items():
        if entity in movie_idx:
            movie_vectors[entity] = vector

    # kNN model
    model = NearestNeighbors()
    model.fit(list(movie_vectors.values()))

    # Predict
    vector = entity_vectors['http://www.wikidata.org/entity/Q5901134']
    distances, indices = model.kneighbors(vector.reshape(1, len(vector)), 10)

    for index in indices.squeeze():
        uri = idx_movie[index]
        print(f'{uri}: {entities[uri]}')


if __name__ == '__main__':
    run()
