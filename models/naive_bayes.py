import numpy as np
from data.training import warm_start

if __name__ == '__main__':
    train, test, n_users, n_movies, n_entities = warm_start(
        ratings_path='../data/mindreader/ratings_clean.json',
        entities_path='../data/mindreader/entities_clean.json',
        conversion_map={
            -1: 1,
            0: None,  # Ignore don't know ratings
            1: 5
        },
        split_ratio=[75, 25]
    )

    n_likes = 0
    n_dislikes = 0
    n_ratings = len(train)

    like_probability = n_likes / n_ratings
    dislike_probability = n_dislikes / n_ratings

    like_probabilities = {}
    dislike_propabilities = {}