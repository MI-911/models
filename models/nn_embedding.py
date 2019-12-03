import json
from random import shuffle, sample

import keras
from keras import Model, Input, Sequential, regularizers
from keras.layers import Dense, Dropout

from data.training import cold_start
import numpy as np

from utilities.util import filter_min_k


def get_model(entity_len, movie_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=entity_len, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(movie_dim, activation='tanh'))

    print(model.summary())

    return model


def min_k(users, k):
    return [(user, ratings) for user, ratings in users if len(ratings['movies']) >= k and len(ratings['entities']) >= k]


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

    # Split users into train and test
    all_users = list(u_r_map.items())
    shuffle(all_users)

    split_index = int(0.75 * len(all_users))
    train_users = all_users[:split_index]
    test_users = all_users[split_index:]

    # Generate training data
    train_x = []
    train_y = []

    for samples in range(1, 11):
        for user, ratings in min_k(train_users, samples):
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
                if y_val[i][true_index] != 1.0:
                    continue

                if true_index in top_k:
                    hits += 1

            print(f'Validation hitrate: {(hits / len(y_val)) * 100}%')

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(np.asarray(train_x), np.asarray(train_y), epochs=50, batch_size=16, verbose=False, validation_split=0.15,
              callbacks=[Metrics()])

    # Smoke test
    while True:
        print(f'Ready for next input')
        inp = input().split(' ')

        smoke_x = np.zeros(len(entity_idx))
        smoke_x[entity_idx[f'http://www.wikidata.org/entity/{inp[0]}']] = inp[1]

        pred = model.predict(np.array([smoke_x])).argsort()[0][::-1][:10]

        for arg in pred:
            predicted_uri = idx_movie[arg]

            print(f'{entities[predicted_uri]} ({predicted_uri})')


if __name__ == '__main__':
    run()
