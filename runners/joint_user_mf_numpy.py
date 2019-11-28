from models.joint_user_mf_numpy import JointUserMF
from data.training import warm_start
import numpy as np
from matplotlib import pyplot as plt
from random import shuffle


def rmse(model, tuples):
    sse = 0
    n = len(tuples)
    for u, m, r, is_movie in tuples:
        sse += ((model.predict(u, m, is_movie) - r) ** 2) / n

    return np.sqrt(sse)


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

    model = JointUserMF(n_users, n_movies, n_entities, k=25, lr=0.0001, reg=0.15)

    # Filter out non-movies?
    train = [(u, m, r, i) for u, m, r, i in train if i]

    n_iter = 100
    eval_every = 10

    train_history = []
    test_history = []

    for i in range(n_iter):
        shuffle(train)
        for u, m, r, is_movie in train:
            model.step(u, m, r, is_movie)

        if i % eval_every == 0:
            train_rmse = rmse(model, train)
            test_rmse = rmse(model, test)
            train_history.append(train_rmse)
            test_history.append(test_rmse)
            print(f'Iteration {i}:')
            print(f'    Train RMSE: {train_rmse}')
            print(f'    Test RMSE:  {test_rmse}')

    plt.plot(train_history, color='orange', label='Train RMSE')
    plt.plot(test_history, color='skyblue', label='Test RMSE')
    plt.legend()
    plt.title(f'Joint User Embedding ({n_iter} iterations)')
    plt.show()
