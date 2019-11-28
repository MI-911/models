import json
from random import shuffle, sample, choice

import numpy as np
from keras import regularizers, Input, Model
from keras.layers import Embedding, Dot, Reshape, Dense, Concatenate, Dropout, merge, Flatten, Multiply
from tensorflow import keras


def load_data(ratings_path='../data/mindreader/user_ratings_map.json'):
    entity_count = 0
    movie_count = 0
    user_count = 0

    entity_idx = dict()
    movie_idx = dict()
    user_idx = dict()

    user_ratings = json.load(open(ratings_path, 'r'))

    # Map movies and entities
    # Skip unknown entities
    for user, ratings in user_ratings.items():
        # Map user
        user_idx[user] = user_count
        user_count += 1

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

    return user_ratings, entity_idx, movie_idx, user_idx


def ratings_to_input(entity_idx, ratings, exclude=None):
    x = np.zeros((len(entity_idx),))

    for item, rating in ratings['entities'] + ratings['movies']: # + ratings['movies']
        # Skip unknown and the item we are predicting
        if not rating or (exclude and item in exclude):
            continue

        x[entity_idx[item]] = rating

    return x


def get_samples(ratings_path='../data/mindreader/user_ratings_map.json'):
    ratings, entity_idx, movie_idx, user_idx = load_data(ratings_path)

    user_ids = []
    item_ids = []
    y = []

    # Go through all users
    for user, ratings in ratings.items():
        rated = [(entity_idx[entity], rating) for entity, rating in ratings['movies'] + ratings['entities'] if rating]

        for entity, rating in rated:
            user_ids.append(user_idx[user])
            item_ids.append(entity)
            y.append(1 if rating == 1 else 0)

    return [item_ids, user_ids], y, entity_idx, user_idx


def get_model(entity_idx, user_idx):
    embedding_size = 100
    layers = [64, 32, 16, 8]

    item = Input(name='item', shape=[1])
    user = Input(name='user', shape=[1])

    mf_item_embedding = Embedding(input_dim=len(entity_idx),
                                  output_dim=embedding_size)

    mf_user_embedding = Embedding(input_dim=len(user_idx),
                                  output_dim=embedding_size)

    nn_item_embedding = Embedding(input_dim=len(entity_idx),
                                  output_dim=embedding_size)

    nn_user_embedding = Embedding(input_dim=len(user_idx),
                                  output_dim=embedding_size)

    # MF part
    mf_user_latent = Flatten()(mf_user_embedding(user))
    mf_item_latent = Flatten()(mf_item_embedding(item))
    mf_vector = Multiply()([mf_user_latent, mf_item_latent])

    # MLP part
    mlp_user_latent = Flatten()(nn_user_embedding(user))
    mlp_item_latent = Flatten()(nn_item_embedding(item))
    mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])
    for idx in range(len(layers)):
        layer = Dense(layers[idx], activation='relu', name="layer%d" % idx)
        mlp_vector = layer(mlp_vector)

    concatenated = Concatenate()([mlp_vector, mf_vector])
    dropout = Dropout(0.5)(concatenated)

    out = Dense(1, activation='sigmoid')(dropout)

    model = Model(inputs=[item, user], outputs=out)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


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
    t_x, t_y, entity_idx, user_idx = get_samples()

    model = get_model(entity_idx, user_idx)
    model.fit(t_x, t_y, batch_size=8, epochs=5, shuffle=True, verbose=True, validation_split=0.15)

    # Generate recommendations
    for_user = user_idx['fake']

    item_ids = np.arange(len(entity_idx))
    user = np.array([for_user for _ in range(len(item_ids))])

    predictions = np.array([prediction[0] for prediction in model.predict([item_ids, user])]).argsort()[::-1][:10]

    idx_entity = {value: key for key, value in entity_idx.items()}

    for prediction in predictions:
        print(idx_entity[prediction])

    print(predictions)


if __name__ == '__main__':
    pred = train()
