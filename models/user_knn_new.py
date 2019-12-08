import json
import random
from collections import defaultdict
from random import sample, choice

import numpy as np
from sklearn.neighbors import NearestNeighbors

from data.training import cold_start
from utilities.metrics import hitrate, ndcg_at_k, average_precision
from utilities.util import get_top_movies, get_entity_occurrence, prune_low_occurrence, split_users


def predict_movies(model, keys, uri_ratings, entity_idx, idx_movie, u_r_map, exclude=None):
    vector = np.zeros(len(entity_idx))

    for uri, rating in uri_ratings:
        vector[entity_idx[uri]] = rating

    distances, indices = model.kneighbors(vector.reshape(1, len(vector)), 15)

    neighbours = list()
    for index, distance in zip(indices.squeeze(), distances.squeeze()):
        neighbours.append((keys[index], distance))

    return _predict(idx_movie, u_r_map, neighbours, exclude)


def _predict(idx_movie, u_r_map, neighbour_weights, exclude=None):
    movie_weight = defaultdict(int)

    for neighbour, weight in neighbour_weights:
        for movie, rating in u_r_map[neighbour]['movies'] + u_r_map[neighbour]['test']:
            movie_weight[idx_movie[movie]] += rating

    # Get weighted prediction and exclude excluded URIs
    predictions = sorted(list(movie_weight.items()), key=lambda x: x[1], reverse=True)
    return [head for head, rating in predictions if not exclude or head not in exclude]


def user_to_vector(ratings, entity_idx, idx_movie):
    vector = np.zeros(len(entity_idx))
    movie_ratings = [(entity_idx[idx_movie[head]], rating) for head, rating in ratings['movies'] + ratings['test']]

    for idx, rating in ratings['entities'] + movie_ratings:
        vector[idx] = rating

    return vector


def run():
    random.seed(42)

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
        split_ratio=[80, 20]
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
    train_users, test_users = split_users(all_users, train_ratio=0.85)

    # Create indices from train users
    user_idx = {}
    user_count = 0
    for user, _ in train_users:
        user_idx[user] = user_count
        user_count += 1

    # Populate user vectors
    user_vectors = {}
    for user, ratings in train_users:
        user_vectors[user] = user_to_vector(ratings, entity_idx, idx_movie)

    # For index lookup
    keys = list(user_vectors.keys())

    # kNN model
    model = NearestNeighbors()
    model.fit(list(user_vectors.values()))

    # Smoke test
    smoke_ratings = [('http://www.wikidata.org/entity/Q3772', like)]

    for prediction in predict_movies(model, keys, smoke_ratings, entity_idx, idx_movie, u_r_map, exclude=None)[:10]:
        print(entities[prediction])

    top_movies = get_top_movies(u_r_map, idx_movie)

    # Metrics
    k = 10
    filtered = test_users
    subsets = {'movies': idx_movie, 'entities': idx_entity, 'popular': None}
    for samples in range(1, 6):
        subset_hits = {subset: 0 for subset in subsets}
        subset_aps = {subset: 0 for subset in subsets}
        subset_ndcg = {subset: 0 for subset in subsets}

        count = 0

        for user, ratings in filtered:
            subset_samples = {}
            skip_user = False

            for subset, idx_lookup in subsets.items():
                if not idx_lookup:
                    continue  # No sampling needed for popular

                sample_from = [(head, rating) for head, rating in ratings[subset] if rating == like]
                if len(sample_from) < samples:
                    skip_user = True
                    break

                subset_samples[subset] = [(idx_lookup[idx], rating) for idx, rating in sample(sample_from, samples)]

            if skip_user or not ratings['test']:
                continue

            ground_truth = [idx_movie[head] for head, rating in ratings['test']]
            left_out = choice(ground_truth)

            # Try both subsets
            subset_predictions = dict()
            for subset, idx_lookup in subsets.items():
                if subset == 'popular':
                    subset_predictions[subset] = top_movies

                    continue

                sampled = subset_samples[subset]

                # Get neighbours and make predictions
                predictions = predict_movies(model, keys, sampled, entity_idx, idx_movie, u_r_map,
                                             exclude=[h for h, _ in sampled])
                subset_predictions[subset] = predictions

            # Metrics
            for subset, predictions in subset_predictions.items():
                subset_aps[subset] += average_precision(ground_truth, predictions, k=k)
                subset_hits[subset] += hitrate(left_out, predictions, k=k)
                subset_ndcg[subset] += ndcg_at_k(ground_truth, predictions, k=k)

            count += 1

        print(f'{samples} samples:')
        for subset in subsets:
            print(f'{subset.title()} MAP: {subset_aps[subset] / count * 100}%')
            print(f'{subset.title()} hit-rate: {subset_hits[subset] / count * 100}%')
            print(f'{subset.title()} NDCG: {subset_ndcg[subset] / count * 100}%')
            print()


if __name__ == '__main__':
    run()
