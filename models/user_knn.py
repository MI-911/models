import json
import random
from collections import defaultdict
from random import shuffle, choice
from scipy import sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from data.training import cold_start
from utilities.util import filter_min_k, get_top_movies, get_entity_occurrence, prune_low_occurrence



def knn(user_vectors, k):
    idx_user = {}
    vectors = []
    for idx, user in enumerate(user_vectors.keys()):
        idx_user[idx] = user
        vectors.append(user_vectors[user])

    similarities = cosine_similarity(sparse.csr_matrix(np.array(vectors)))

    all_similarities = dict()

    # For each user, get the top k nearest neighbours
    for idx in idx_user.keys():
        # Get the users that are nearest to us (skip the first element, that's us)
        similarity_values = np.delete(similarities[idx], idx)

        sorted_args = np.argsort(similarity_values)[::-1]

        all_similarities[idx_user[idx]] = [(idx_user[idx], similarity_values[idx]) for idx in sorted_args[:k]]

    return all_similarities


def predict_movies(idx_movie, u_r_map, neighbour_weights, exclude=None):
    movie_weight = defaultdict(int)

    for neighbour, weight in neighbour_weights:
        for movie, rating in u_r_map[neighbour]['movies']:
            movie_weight[idx_movie[movie]] += rating

    # Get weighted prediction and exclude excluded URIs
    predictions = sorted(list(movie_weight.items()), key=lambda x: x[1], reverse=True)
    return [head for head, rating in predictions if not exclude or head not in exclude]


def run():
    dislike = None
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
    u_r_map = prune_low_occurrence(u_r_map, idx_entity, idx_movie, entity_occurrence, keep=4)
    filtered = filter_min_k(u_r_map, 10).items()

    # Take one test item from each user
    for user, ratings in filtered:
        test_head = choice(ratings['movies'])[0]

        ratings['test'] = idx_movie[test_head]
        ratings['movies'] = [(head, rating) for head, rating in ratings['movies'] if head != test_head]

    # Construct user vectors
    user_vectors = {}
    subsets = {'movies': idx_movie, 'entities': idx_entity, 'popular': None}
    for user, ratings in filtered:
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

    for neighbours in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800]:
        print(f'{neighbours=}')

        count = 0
        hits = 0
        pop_hits = 0

        weights = knn(user_vectors, neighbours)
        for user, ratings in filtered:
            own = [head for head, rating in ratings['movies']]
            predictions = predict_movies(idx_movie, u_r_map, weights[user], exclude=own)

            if ratings['test'] in predictions[:k]:
                hits += 1

            if ratings['test'] in top_movies[:k]:
                pop_hits += 1

            count += 1

        print(f'UserKNN Hits: {hits / count * 100}%')
        print(f'TopPop Hits: {pop_hits / count * 100}%')


if __name__ == '__main__':
    random.seed(4)
    run()
