import json
from random import shuffle, sample, choice

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


def ratings_to_input(entity_idx, ratings, exclude=None):
    x = np.zeros((len(entity_idx),))

    for item, rating in ratings['entities'] + ratings['movies']: # + ratings['movies']
        # Skip unknown and the item we are predicting
        if not rating or (exclude and item in exclude):
            continue

        x[entity_idx[item]] = rating

    return x


def get_samples(ratings_path='../data/mindreader/user_ratings_map.json'):
    user_ratings, entity_idx, movie_idx = load_data(ratings_path)

    all_user_ratings = list(user_ratings.items())
    shuffle(all_user_ratings)

    split_index = int(len(all_user_ratings) * 0.85)

    train_user_ratings = all_user_ratings[:split_index]
    test_user_ratings = all_user_ratings[split_index:]

    train_x, train_y = [list() for _ in range(2)]

    # Go through all users
    for user, ratings in train_user_ratings:
        liked_movies = [(movie, rating) for movie, rating in ratings['movies'] if rating]
        if not liked_movies:
            continue

        # Randomly select half for prediction
        shuffle(liked_movies)
        # rated_movies = rated_movies[len(rated_movies) // 2:]

        # For each liked movie, create a training sample which tries to learn that movie
        y = np.zeros((len(movie_idx),))
        for liked_movie, rating in liked_movies:
            y[movie_idx[liked_movie]] = rating

        # Consider all other liked or disliked entities as input
        train_x.append(ratings_to_input(entity_idx, ratings, exclude=[movie for movie, _ in liked_movies]))
        train_y.append(y)

    return np.array(train_x), np.array(train_y), entity_idx, movie_idx, test_user_ratings


def test(model, test_user_ratings, entity_idx, movie_idx, k=5):
    hits = 0

    for user, ratings in test_user_ratings:
        # Sample a movie to guess from the user's remaining information
        movie = choice([item for item, rating in ratings['movies'] if item in movie_idx])

        # Note that the sampled movie is not used in the prediction
        x = ratings_to_input(entity_idx, ratings, exclude=[movie])

        top_k = model.predict(np.array([x]))[0].argsort()[::-1][:k]

        # Check if the sampled movie's index is in top k
        if movie_idx[movie] in top_k:
            hits += 1

    print(f'Final hitrate: {(hits / len(test_user_ratings)) * 100}%')


def train():
    from keras.models import Sequential
    from keras.layers import Dense, Dropout

    t_x, t_y, entity_idx, movie_idx, test_user_ratings = get_samples()

    model = Sequential()
    model.add(Dense(4096, input_dim=t_x.shape[1]))
    model.add(Dense(2048, activation='sigmoid'))
    model.add(Dense(512))
    model.add(Dense(t_y.shape[1], activation='tanh'))

    model.compile(optimizer='adam', loss='mean_squared_error')

    class Metrics(keras.callbacks.Callback):
        def on_epoch_end(self, batch, logs=None):
            X_val, y_val = self.validation_data[0], self.validation_data[1]
            y_predict = np.asarray(model.predict(X_val))

            hits = 0
            for i in range(len(y_val)):
                top_k = y_predict[i].argsort()[::-1][:5]
                true_index = y_val[i].argmax()

                if true_index in top_k:
                    hits += 1

            print(f'Hitrate: {(hits / len(y_val)) * 100}%')

    model.fit(t_x, t_y, epochs=500, batch_size=8, verbose=True, validation_split=0.15, callbacks=[Metrics()])

    # Synthetic user
    test_x = np.zeros((len(entity_idx),))
    test_x[entity_idx['http://www.wikidata.org/entity/Q3772']] = 1

    pred = model.predict(np.array([test_x])).argsort()[0][::-1][:10]

    idx_movie = {value: key for key, value in movie_idx.items()}

    for arg in pred:
        print(idx_movie[arg])

    # Test
    test(model, test_user_ratings, entity_idx, movie_idx)


if __name__ == '__main__':
    pred = train()
