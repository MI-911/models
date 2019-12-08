import json
import random
from collections import defaultdict

import numpy as np
from sklearn.neighbors import NearestNeighbors

from data.training import cold_start
from utilities.util import get_entity_occurrence, prune_low_occurrence, split_users


def similarity(a, b):
    return np.linalg.norm(a) - np.linalg.norm(b)


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


def predict(model, idx_movie, entity_vectors, uris, neighbors=10):
    weights = defaultdict(int)

    for uri in uris:
        vector = entity_vectors[uri]
        distances, indices = model.kneighbors(vector.reshape(1, len(vector)), neighbors + 1)

        # Get URI predictions
        uri_predictions = {idx_movie[idx]: distance for idx, distance in zip(indices.squeeze(), distances.squeeze())}

        # Remove duplicates
        uri_predictions = list({key: value for key, value in uri_predictions.items() if key not in uris}.items())

        # Sort and limit to n neighbours
        # Should not be reverse sorted since it's by distance
        uri_predictions = sorted(uri_predictions, key=lambda item: item[1])[:neighbors]

        # Add weights
        for target, weight in uri_predictions:
            weights[target] += weight

    return [uri for uri, rating in sorted(weights.items(), key=lambda item: item[1], reverse=False)]


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
    item_variances = get_variances(entity_vectors)
    json.dump(item_variances, open('variances.json', 'w'))

    # Get movie vectors
    movie_vectors = dict()
    for entity, vector in entity_vectors.items():
        if entity in movie_idx:
            movie_vectors[entity] = vector

    # kNN model
    model = NearestNeighbors(metric='cosine')
    model.fit(list(movie_vectors.values()))

    prediction = predict(model, idx_movie, entity_vectors, ['http://www.wikidata.org/entity/Q200092', 'http://www.wikidata.org/entity/Q157394'])[:10]
    for pred in prediction:
        print(entities[pred])


if __name__ == '__main__':
    random.seed(42)

    run()
