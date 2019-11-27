from data.training import warm_start
from models.joint_user_mf_torch import JointUserMF
import torch.optim as optimizers
import torch.nn as nn
import torch as tt
from random import shuffle
from matplotlib import pyplot as plt


def to_tensors(users, movies, ratings, is_movies):
    return to_long(users), to_long(movies), to_float(ratings), to_float(is_movies).view((len(is_movies), 1))


def to_float(val):
    return tt.tensor(val).to(tt.float)


def to_long(val):
    return tt.tensor(val).to(tt.long)


def rmse(predictions, ys):
    return tt.sqrt(nn.MSELoss()(predictions, ys))


def batches(lst, n=1):
    l = len(lst)
    for ndx in range(0, l, n):
        batch = lst[ndx:min(ndx + n, l)]
        users, movies, ratings, is_movies = zip(*batch)
        yield to_tensors(users, movies, ratings, is_movies)


def unpack_user_ratings(u_r_map, n_movies):
    train = []
    test = []

    for u, ratings in u_r_map.items():
        for m, r in ratings['movies']:
            train.append((u, m, r, 1))
        for e, r in ratings['entities']:
            train.append((u, e + n_movies, r, 0))
        for m, r in ratings['test']:
            test.append((u, m, r, 1))

    shuffle(train)
    shuffle(test)

    return train, test


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

    n_iter = 1000
    k = 10
    lr = 0.001
    eval_every = 10

    model = JointUserMF(n_users=n_users, n_movies=n_movies, n_entities=n_entities, k=k)
    optimizer = optimizers.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_history = []
    test_history = []

    for i in range(n_iter):
        for users, movies, ratings, is_movie in batches(train, n=64):
            model.train()
            predictions = model(users, movies, is_movie)
            loss = loss_fn(predictions, tt.tensor(ratings))
            loss.backward()
            optimizer.step()

            model.zero_grad()

        if i % eval_every == 0:
            with tt.no_grad():
                model.eval()
                # Calculate full train and test loss
                train_users, train_movies, train_ratings, train_is_movies = to_tensors(*zip(*train))
                test_users, test_movies, test_ratings, test_is_movies = to_tensors(*zip(*test))

                train_predictions = model(train_users, train_movies, train_is_movies)
                test_predictions = model(test_users, test_movies, test_is_movies)

                train_loss = rmse(train_predictions, train_ratings)
                test_loss = rmse(test_predictions, test_ratings)

                train_history.append(train_loss)
                test_history.append(test_loss)

                print(f'Iteration {i}:')
                print(f'    Train RMSE: {train_loss}')
                print(f'    Test  RMSE: {test_loss}')

    plt.plot(train_history, color='orange', label='Train RMSE')
    plt.plot(test_history, color='skyblue', label='Test RMSE')
    plt.legend()
    plt.title(f'Joint User Embedding ({n_iter} iterations)')
    plt.show()


