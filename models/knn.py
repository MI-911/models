from random import sample, shuffle, choice

from data.training import cold_start
from scipy import spatial

from models.pagerank import filter_min_k, get_relevance_list, average_precision
import numpy as np


def similarity(a, b):
    return a.dot(b)


def knn(user_vectors, user, own_vector):
    similarities = []

    for other, other_vector in user_vectors.items():
        if other == user:
            continue

        similarities.append((other, similarity(own_vector, other_vector)))

    # Shuffle s.t. any secondary ordering is random
    shuffle(similarities)

    return sorted(similarities, key=lambda x: x[1], reverse=True)[:5]


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
            0: None,  # Ignore don't know ratings
            1: 1
        },
        restrict_entities=None,
        split_ratio=[80, 20]
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

    subsets = {'movies': idx_movie, 'entities': idx_entity}
    # Construct user vectors
    user_vectors = {}
    for user, ratings in u_r_map.items():
        user_vectors[user] = np.zeros(len(entity_idx))

        for subset, idx_lookup in subsets.items():
            sampled = [(idx_lookup[idx], rating) for idx, rating in ratings[subset]]

            for uri, rating in sampled:
                uri_idx = entity_idx[uri]

                user_vectors[user][uri_idx] = rating

    # Sample k movies and entities
    for samples in range(1, 11):
        subset_hits = {subset: 0 for subset in subsets}
        subset_aps = {subset: 0 for subset in subsets}
        count = 0

        for user, ratings in filter_min_k(u_r_map, samples).items():
            ground_truth = [idx_movie[head] for head, rating in ratings['test'] if rating == 1]
            if not ground_truth:
                continue

            left_out = choice(ground_truth)

            # Try both subsets
            for subset, idx_lookup in subsets.items():
                sampled = [(idx_lookup[idx], rating) for idx, rating in sample(ratings[subset], samples)]

                own_vector = np.zeros(len(entity_idx))
                for uri, rating in sampled:
                    own_vector[entity_idx[uri]] = rating

                # Get neighbours and make predictions
                neighbour_weights = knn(user_vectors, user, own_vector)
                predictions = predict_movies(idx_movie, u_r_map, neighbour_weights, exclude=[h for h, _ in sampled])

                # Add to metrics
                subset_aps[subset] += average_precision(ground_truth, predictions)
                subset_hits[subset] += 1 if left_out in predictions[:10] else 0

            count += 1

        print(f'{samples} samples:')
        for subset in subsets:
            print(f'{subset.title()} MAP: {subset_aps[subset] / count}')
            print(f'{subset.title()} hit-rate: {subset_hits[subset] / count}')
            print()


if __name__ == '__main__':
    run()
