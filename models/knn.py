from random import sample, shuffle, choice
import itertools
from data.training import cold_start
from scipy import spatial

import numpy as np

from utilities.metrics import ndcg_at_k, average_precision, hitrate
from utilities.util import filter_min_k, get_top_movies


def similarity(a, b):
    return a.dot(b)


def knn(user_vectors, user, own_vector, neighbours):
    similarities = []

    for other, other_vector in user_vectors.items():
        if other == user:
            continue

        similarities.append((other, similarity(own_vector, other_vector)))

    # Shuffle s.t. any secondary ordering is random
    shuffle(similarities)

    return sorted(similarities, key=lambda x: x[1], reverse=True)[:neighbours]


def predict_movies(idx_movie, u_r_map, neighbour_weights, exclude=None):
    movie_weight = dict()

    for neighbour, weight in neighbour_weights:
        for movie, rating in u_r_map[neighbour]['movies']:
            movie_uri = idx_movie[movie]
            movie_weight[movie_uri] = movie_weight.get(movie_uri, 0) + rating * weight

    # Get weighted prediction and exclude excluded URIs
    predictions = sorted(list(movie_weight.items()), key=lambda x: x[1], reverse=True)
    return [head for head, rating in predictions if not exclude or head not in exclude]


def run():
    u_r_map, n_users, movie_idx, entity_idx = cold_start(
        from_path='../data/mindreader/user_ratings_map.json',
        conversion_map={
            -1: -1,
            0: None,
            1: 1
        },
        restrict_entities=None,
        split_ratio=[100, 0]
    )

    # Add movies to the entity_idx map
    # Just makes it easier when constructing user vectors
    entity_count = len(entity_idx)
    for movie, idx in movie_idx.items():
        if movie not in entity_idx:
            entity_idx[movie] = entity_count
            entity_count += 1

    idx_entity = {value: key for key, value in entity_idx.items()}
    idx_movie = {value: key for key, value in movie_idx.items()}

    subsets = {'movies': idx_movie, 'entities': idx_entity, 'popular': None}
    # Construct user vectors
    user_vectors = {}
    for user, ratings in u_r_map.items():
        user_vectors[user] = np.zeros(len(entity_idx))

        for subset, idx_lookup in subsets.items():
            if not idx_lookup:
                continue

            sampled = [(idx_lookup[idx], rating) for idx, rating in ratings[subset]]

            for uri, rating in sampled:
                user_vectors[user][entity_idx[uri]] = rating

    # Static, non-personalized measure of top movies
    top_movies = get_top_movies(u_r_map, idx_movie)

    # Sample k movies and entities
    k = 10
    for samples in range(1, 11):
        subset_hits = {subset: 0 for subset in subsets}
        subset_aps = {subset: 0 for subset in subsets}
        subset_ndcg = {subset: 0 for subset in subsets}

        count = 0

        for user, ratings in filter_min_k(u_r_map, samples).items():
            subset_samples = {}
            for subset, idx_lookup in subsets.items():
                if idx_lookup:
                    subset_samples[subset] = [(idx_lookup[idx], rating) for idx, rating in sample(ratings[subset], samples)]

            all_samples = set(itertools.chain.from_iterable(subset_samples.values()))

            ground_truth = [idx_movie[head] for head, rating in ratings['movies'] if rating == 1]
            ground_truth = [head for head in ground_truth if head not in all_samples]
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
                neighbour_weights = knn(user_vectors, user, own_vector, neighbours=15)
                predictions = predict_movies(idx_movie, u_r_map, neighbour_weights, exclude=[h for h, _ in sampled])
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
