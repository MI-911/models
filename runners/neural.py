import json
from random import shuffle

import numpy as np
from keras import regularizers
from tensorflow import keras


def load_data(ratings_path='../data/mindreader/user_ratings_map.json'):
    entity_count = 0
    movie_count = 0
    entity_idx = dict()
    movie_idx = dict()

    user_ratings = json.load(open(ratings_path, 'r'))

    # Map movies and entities
    # Skip unknown entities
    for user, ratings in user_ratings.items():
        # Map movies
        for movie in [movie for movie, rating in ratings['movies'] if rating]:
            if movie not in movie_idx:
                movie_idx[movie] = movie_count
                movie_count += 1

            if movie not in entity_idx:
                entity_idx[movie] = entity_count
                entity_count += 1

        # Map entities
        for entity in [entity for entity, rating in ratings['entities'] if rating]:
            if entity not in entity_idx:
                entity_idx[entity] = entity_count
                entity_count += 1

    return user_ratings, entity_idx, movie_idx


def get_samples(ratings_path='../data/mindreader/user_ratings_map.json'):
    user_ratings, entity_idx, movie_idx = load_data(ratings_path)

    train_x, train_y = [list() for _ in range(2)]

    # Go through all users
    for user, ratings in user_ratings.items():
        rated_movies = [(movie, rating) for movie, rating in ratings['movies'] if rating]
        if not rated_movies:
            return

        # Randomly select half for prediction
        shuffle(rated_movies)
        # rated_movies = rated_movies[len(rated_movies) // 2:]

        # For each liked movie, create a training sample which tries to learn that movie
        y = np.zeros((len(movie_idx),))
        for liked_movie, rating in rated_movies:
            y[movie_idx[liked_movie]] = rating

        # Consider all other liked or disliked entities as input
        x = np.zeros((len(entity_idx),))
        for item, rating in ratings['movies'] + ratings['entities']:
            # Skip unknown and the item we are predicting
            if not rating or item in rated_movies:
                continue

            x[entity_idx[item]] = rating

        train_x.append(x)
        train_y.append(y)

    return np.array(train_x), np.array(train_y), entity_idx, movie_idx


def train():
    from keras.models import Sequential
    from keras.layers import Dense, Dropout

    t_x, t_y, entity_idx, movie_idx = get_samples()

    model = Sequential()
    model.add(Dense(256, input_dim=t_x.shape[1]))
    model.add(Dense(512))
    model.add(Dropout(0.15))
    model.add(Dense(t_y.shape[1], activation='tanh'))

    model.compile(optimizer='adam', loss='mean_squared_error')

    class Metrics(keras.callbacks.Callback):
        def on_epoch_end(self, batch, logs=None):
            X_val, y_val = self.validation_data[0], self.validation_data[1]
            y_predict = np.asarray(model.predict(X_val))

            hits = 0
            for i in range(len(y_val)):
                top_k = y_predict[i].argsort()[::-1][:10]
                true_index = y_val[i].argmax()

                if true_index in top_k:
                    hits += 1

            print(f'Hitrate: {(hits / len(y_val)) * 100}%')

    model.fit(t_x, t_y, epochs=200, batch_size=32, verbose=False, validation_split=0.1, callbacks=[Metrics()])

    # Predict with Tom Hanks
    test_x = np.zeros((len(entity_idx),))
    test_x[entity_idx['http://www.wikidata.org/entity/Q842256']] = 1

    pred = model.predict(np.array([test_x])).argsort()[0][::-1][:10]
    #return pred

    idx_movie = {value: key for key, value in movie_idx.items()}

    for arg in pred:
        print(idx_movie[arg])


if __name__ == '__main__':
    pred = train()
