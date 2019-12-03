import json
import random
from random import shuffle, sample, choice

import keras
from keras import Model, Input, Sequential, regularizers
from keras.layers import Dense, Dropout

from data.training import cold_start
import numpy as np

from utilities.metrics import average_precision, hitrate, ndcg_at_k
from utilities.util import filter_min_k, get_top_movies, get_entity_occurrence, prune_low_occurrence


def get_model(entity_len, movie_dim):
    model = Sequential()

    model.add(Dense(32, input_dim=entity_len, activation='tanh'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(movie_dim, activation='tanh'))

    print(model.summary())

    return model


def min_k(users, k):
    return [(user, ratings) for user, ratings in users if len(ratings['movies']) >= k and len(ratings['entities']) >= k]


def run():
    like_signal = 1
    dislike_signal = -1
    unknown_signal = None

    u_r_map, n_users, movie_idx, entity_idx = cold_start(
        conversion_map={
            -1: dislike_signal,
            0: unknown_signal,
            1: like_signal
        },
        restrict_entities=None,
        split_ratio=[75, 25]
    )

    # Get entities
    with open('../data/mindreader/entities_clean.json') as fp:
        entities = dict()

        for uri, name, labels in json.load(fp):
            entities[uri] = name

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

    # Generate training data
    train_x = []
    train_y = []

    for samples in range(1, 21):
        for user, ratings in filter_min_k(u_r_map, samples).items():
            sampled_entities = sample(ratings['entities'], samples)
            sampled_movies = sample(ratings['movies'], samples)
            predict = [(movie, rating) for movie, rating in ratings['movies'] if (movie, rating) not in sampled_movies]

            # Only create target if there is anything to predict
            if not predict:
                continue

            # Create target
            y = np.zeros(len(movie_idx))
            for idx, rating in predict:
                y[idx] = rating

            # Create input
            lookups = {
                'movies': {
                    'samples': sampled_movies,
                    'lookup': idx_movie
                },
                'entities': {
                    'samples': sampled_entities,
                    'lookup': idx_entity
                }
            }

            x = np.zeros(len(entity_idx))
            for category, lookup in lookups.items():
                for sample_idx, rating in lookup['samples']:
                    idx = entity_idx[lookup['lookup'][sample_idx]]

                    x[idx] = rating

            train_x.append(x)
            train_y.append(y)

    # Train model
    model = get_model(len(entity_idx), len(movie_idx))

    class Metrics(keras.callbacks.Callback):
        def on_epoch_end(self, batch, logs=None):
            x_val, y_val = self.validation_data[0], self.validation_data[1]
            y_predict = np.asarray(model.predict(x_val))

            hits = 0
            for i in range(len(y_val)):
                top_k = y_predict[i].argsort()[::-1][:5]
                true_index = y_val[i].argmax()
                if y_val[i][true_index] != like_signal:
                    continue

                if true_index in top_k:
                    hits += 1

            print(f'Validation hitrate: {(hits / len(y_val)) * 100}%')

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(np.asarray(train_x), np.asarray(train_y), epochs=50, batch_size=8, verbose=False, validation_split=0.15,
              callbacks=[Metrics()])

    # Static, non-personalized measure of top movies
    top_movies = get_top_movies(u_r_map, idx_movie)

    # Test with different samples
    subsets = {'entities': idx_entity, 'movies': idx_movie, 'popular': None}
    k = 10
    for samples in range(1, 6):
        filtered = filter_min_k(u_r_map, samples)
        if not filtered:
            break

        subset_hits = {subset: 0 for subset in subsets}
        subset_aps = {subset: 0 for subset in subsets}
        subset_ndcg = {subset: 0 for subset in subsets}
        count = 0

        for user, ratings in filtered.items():
            ground_truth = [idx_movie[idx] for idx, rating in ratings['test'] if rating == like_signal]
            if not ground_truth:
                continue

            left_out = choice(ground_truth)

            subset_samples = {subset: sample(ratings[subset], samples) for subset in subsets if subset != 'popular'}

            # Try both subsets
            subset_predictions = dict()
            for subset, idx_lookup in subsets.items():
                if subset == 'popular':
                    subset_predictions[subset] = top_movies

                    continue

                sampled = subset_samples[subset]

                # Input vector from sampled
                x = np.zeros(len(idx_entity))
                for idx, rating in sampled:
                    uri = subsets[subset][idx]
                    x[entity_idx[uri]] = rating

                prediction = [idx_movie[pred_idx] for pred_idx in model.predict(np.array([x])).argsort()[0][::-1]]

                subset_predictions[subset] = prediction

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
