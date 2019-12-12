import json
import random
from collections import defaultdict

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

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


def new_pred(model, idx_movies, idx_descriptive_entities, entity_vectors, item, k=10):
    vector = entity_vectors[item]
    continue_increase = True
    increase_val = k*10
    while continue_increase:
        distances, indices = model.kneighbors(vector.reshape(1, len(vector)), k * increase_val)
        distances, indices = distances[0], indices[0]

        num_movies = len([1 for i in indices if i in idx_movies])
        num_entities = len([1 for i in indices if i in idx_descriptive_entities])

        if num_movies < k and num_entities < k:
            increase_val += k
        else:
            continue_increase = False

    top_m = [(d, i) for d, i in zip(distances, indices) if i in idx_movies][:k]
    top_de = [(d, i) for d, i in zip(distances, indices) if i in idx_descriptive_entities][:k]

    # TODO: include user and item biases, maybe normalise.
    rating_m = sum([d * idx_movies[m] for d, m in top_m])
    rating_de = sum([d * idx_descriptive_entities[de] for d, de in top_de])

    return rating_m, rating_de


def split_data(u_r_map, movies):
    train, test = [], []
    for user, ratings in u_r_map.items():
        sample = random.sample(ratings['movies'], 1)[0]
        ratings['movies'].remove(sample)
        train.append((user, ratings))
        test.append((user, sample))

    return train, test


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
    train_users, test_users = split_data(u_r_map, idx_movie)

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

    prediction = predict(model, idx_movie, entity_vectors, ['http://www.wikidata.org/entity/Q200092'])[:10]
    for pred in prediction:
        print(entities[pred])

    # single_sample(test_users, idx_entity, idx_movie, model, entity_vectors)
    # all_combinations(test_users, idx_entity, idx_movie, model, entity_vectors)
    simple_hitrate(train_users, test_users, idx_entity, idx_movie, model, entity_vectors, 10)


def single_sample(test_users, idx_entity, idx_movie, model, entity_vectors):
    m_hits = 0
    e_hits = 0
    count = 0

    # Go through all users in test set
    for user, ratings in tqdm(test_users):
        # Look only on liked items
        entity_ratings = [idx_entity[head] for head, rating in ratings['entities'] if rating == 1]
        movie_ratings = [idx_movie[head] for head, rating in ratings['movies'] if rating == 1]

        # Shuffle to take sample randomly
        random.shuffle(entity_ratings)
        random.shuffle(movie_ratings)

        entity_sample = entity_ratings[0]
        movie_sample, movie_sample_true = movie_ratings[:2]

        # Get Top k predictions
        e_pred = predict(model, idx_movie, entity_vectors, [entity_sample])[:10]
        m_pred = predict(model, idx_movie, entity_vectors, [movie_sample])[:10]

        # Check if sampled movie in pred
        if movie_sample_true in e_pred:
            e_hits += 1
        if movie_sample_true in m_pred:
            m_hits += 1

        count += 1

    print(f'Movie hitrate: {m_hits / count}')
    print(f'Entities hitrate {e_hits / count}')


def all_combinations(test_users, idx_entity, idx_movie, model, entity_vectors):
    m_hits = 0
    e_hits = 0
    m_c = 0
    e_c = 0

    # Go through all users in test set
    for user, ratings in tqdm(test_users):
        # Look only on liked items
        entity_ratings = [idx_entity[head] for head, rating in ratings['entities'] if rating == 1]
        movie_ratings = [idx_movie[head] for head, rating in ratings['movies'] if rating == 1]

        num_samples = min(len(entity_ratings), len(movie_ratings))
        entity_samples = random.sample(entity_ratings, num_samples)
        movie_samples = random.sample(movie_ratings, num_samples)
        for sample in entity_samples:
            # Get Top k predictions
            pred = predict(model, idx_movie, entity_vectors, [sample])[:10]

            # Find how many movies are in Top k
            for m_r in movie_ratings:
                if m_r in pred:
                    e_hits += 1
                e_c += 1

        for sample in movie_samples:
            # Get Top k predictions
            pred = predict(model, idx_movie, entity_vectors, [sample])[:10]

            # Find how many movies are in Top k
            for m_r in movie_ratings:
                if m_r != sample and m_r in pred:
                    m_hits += 1

                m_c += 1

    print(f'Movie hitrate: {m_hits / m_c}')
    print(f'Entities hitrate {e_hits / e_c}')


def simple_hitrate(train, test, idx_entity, idx_movie, model, entity_vectors, k):
    idx_train = {u: rs for u, rs in train}
    movie_ids = list(idx_movie.keys())

    e_hits, m_hits = 0, 0
    count = 0

    iter_num = 0
    # Go through all users in test set
    for user, (movie_id, rating) in test:
        iter_num += 1
        print(f'{iter_num}/{len(test)}')
        a = idx_train[user]
        user_idx_movie = {u: r for u, r in idx_train[user]['movies']}
        user_idx_entity = {u: r for u, r in idx_train[user]['entities']}

        # Only use users with at least k movie and entity ratings.
        if len(user_idx_movie) < k or len(user_idx_entity) < k:
            continue

        # Sample 1000 movies + test uri.
        random.shuffle(movie_ids)
        samples = list(filter(lambda x: (x not in user_idx_movie) and x != movie_id, movie_ids))[:500]
        samples.append(movie_id)
        sample_len = len(samples)

        # Find rating for all sampled movies
        movie_ratings = []
        entity_ratings = []
        for i, sample in tqdm(enumerate(samples), total=sample_len, desc='inner', position=0):
            sample_uri = idx_movie[sample] if sample in idx_movie else idx_entity[sample]
            movie_rating, entity_rating = new_pred(model, user_idx_movie, user_idx_entity, entity_vectors, sample_uri, k)

            movie_ratings.append((i, movie_rating))
            entity_ratings.append((i, entity_rating))

        # Sort movies and take top k
        movie_ratings = sorted(movie_ratings, key=lambda x: x[1], reverse=True)[:k]
        entity_ratings = sorted(entity_ratings, key=lambda x: x[1], reverse=True)[:k]

        # Check if test movie in top k of movie and entity ratings
        for i, _ in movie_ratings:
            if i == sample_len-1:
                m_hits += 1

        for i, _ in entity_ratings:
            if i == sample_len-1:
                e_hits += 1

        count += 1

        print(f'Movie hitrate: {m_hits / count}')
        print(f'Entities hitrate {e_hits / count}')

    print(f'Movie hitrate: {m_hits / count}')
    print(f'Entities hitrate {e_hits / count}')


if __name__ == '__main__':
    random.seed(42)

    run()
