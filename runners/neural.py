import json
import numpy as np
import keras
from keras import Input


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
        liked_movies = [movie for movie, rating in ratings['movies'] if rating == 1]

        # For each liked movie, create a training sample which tries to learn that movie
        for liked_movie in liked_movies:
            # One hot vector of the predicted movie
            y = np.zeros((len(movie_idx),))
            y[movie_idx[liked_movie]] = 1

            # Consider all other liked or disliked entities as input
            x = np.zeros((len(entity_idx),))
            for item, rating in ratings['movies'] + ratings['entities']:
                # Skip unknown and the item we are predicting
                if rating == 0 or item == liked_movie:
                    continue

                x[entity_idx[item]] = rating

            train_x.append(x)
            train_y.append(y)

    return np.array(train_x), np.array(train_y), entity_idx, movie_idx


def train():
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout

    t_x, t_y, entity_idx, movie_idx = get_samples()

    print(t_x.shape[1])

    model = Sequential()
    model.add(Dense(128, input_dim=t_x.shape[1]))
    model.add(Dropout(rate=.25))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(t_y.shape[1], activation='sigmoid'))

    model.compile(optimizer='adam', loss='categorical_crossentropy',)
    model.fit(t_x, t_y, epochs=50, batch_size=16, verbose=False, validation_split=0.25)

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

