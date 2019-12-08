import json
import random
from collections import defaultdict
from random import sample, shuffle, choice
import itertools

import tqdm

from data.training import cold_start
from scipy import spatial

import numpy as np

from utilities.metrics import ndcg_at_k, average_precision, hitrate
from utilities.util import filter_min_k, get_top_movies, get_entity_occurrence, prune_low_occurrence


def similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


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
            movie_weight[idx_movie[movie]] += rating * weight

    # Get weighted prediction and exclude excluded URIs
    predictions = sorted(list(movie_weight.items()), key=lambda x: x[1], reverse=True)
    return [head for head, rating in predictions if not exclude or head not in exclude]


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
        split_ratio=[80, 20]
    )

    # Load variances
    variances = json.load(open('../data/mindreader/variances.json'))

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

    # Construct user vectors
    user_vectors = {}
    subsets = {'movies': idx_movie, 'entities': idx_entity, 'popular': None}
    for user, ratings in u_r_map.items():
        user_vectors[user] = np.zeros(len(entity_idx))

        for subset, idx_lookup in subsets.items():
            if not idx_lookup:
                continue

            sampled = [(idx_lookup[idx], rating) for idx, rating in ratings[subset]]

            for uri, rating in sampled:
                user_vectors[user][entity_idx[uri]] = rating

    # k is the number of items to evaluate in the top
    k = 10

    # Static, non-personalized measure of top movies
    top_movies = get_top_movies(u_r_map, idx_movie)

    filtered = filter_min_k(u_r_map, 10).items()
    neighbours = 30
    for samples in range(4, 6):
        subset_hits = {subset: 0 for subset in subsets}
        subset_aps = {subset: 0 for subset in subsets}
        subset_ndcg = {subset: 0 for subset in subsets}

        count = 0

        for user, ratings in tqdm.tqdm(filtered):
            subset_samples = {}
            for subset, idx_lookup in subsets.items():
                if idx_lookup:
                    subset_ratings = []

                    for idx, rating in ratings[subset]:
                        uri = idx_lookup[idx]
                        subset_ratings.append((uri, rating, variances[uri]))

                    # Get top by variance
                    subset_ratings = sorted(subset_ratings, key=lambda item: item[2], reverse=True)[:samples]

                    subset_samples[subset] = [(uri, rating) for uri, rating, _ in subset_ratings]

            ground_truth = [idx_movie[head] for head, rating in ratings['test']]
            if not ground_truth:
                continue

            left_out = choice(ground_truth)

            # Try both subsets
            subset_predictions = dict()
            for subset, idx_lookup in subsets.items():
                if subset == 'popular':
                    subset_predictions[subset] = top_movies

                    continue

                sampled = subset_samples[subset]

                own_vector = np.zeros(len(entity_idx))
                for uri, rating in sampled:
                    own_vector[entity_idx[uri]] = rating

                # Get neighbours and make predictions
                neighbour_weights = knn(user_vectors, user, own_vector, neighbours=neighbours)
                predictions = predict_movies(idx_movie, u_r_map, neighbour_weights, top_movies[:k],
                                             popularity_bias=0, exclude=[h for h, _ in sampled])
                subset_predictions[subset] = predictions

            # Metrics
            for subset, predictions in subset_predictions.items():
                subset_aps[subset] += average_precision(ground_truth, predictions, k=k)
                subset_hits[subset] += hitrate(left_out, predictions, k=k)
                subset_ndcg[subset] += ndcg_at_k(ground_truth, predictions, k=k)

            count += 1

        print(f'{samples} samples, {neighbours} neighbours:')
        for subset in subsets:
            print(f'{subset.title()} MAP: {subset_aps[subset] / count * 100}%')
            print(f'{subset.title()} hit-rate: {subset_hits[subset] / count * 100}%')
            print(f'{subset.title()} NDCG: {subset_ndcg[subset] / count * 100}%')
            print()


if __name__ == '__main__':
    random.seed(1)
    run()
