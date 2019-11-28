import torch as tt
import torch.optim as optimizers
from data.training import cold_start
from models.all_nn import NeuralNetworkAll
import torch.nn as nn
from random import shuffle


def unify_indeces(u_r_map, n_movies):
    _map = {}
    for u, ratings in u_r_map.items():
        movie_ratings = []
        entity_ratings = []
        test_ratings = []
        for m, r in ratings['movies']:
            movie_ratings.append((m, r))
        for e, r in ratings['entities']:
            entity_ratings.append((e + n_movies, r))
        for m, r in ratings['test']:
            test_ratings.append((m, r))

        _map[u] = {
            'movies': movie_ratings,
            'entities': entity_ratings,
            'test': test_ratings
        }

    return _map


def one_hot_samples(u_r_map, n_movies, n_entities, train=True):
    samples = []
    labels = []
    for u, ratings in u_r_map.items():
        x = tt.zeros(n_movies + n_entities).to(tt.float)
        y = tt.zeros(n_movies + n_entities).to(tt.long)

        entity_split_idx = int(len(ratings['entities']) * 0.5)
        entity_ratings = ratings['entities']
        shuffle(entity_ratings)
        entity_ratings = entity_ratings[:entity_split_idx] if train else entity_ratings[entity_split_idx:]

        m_idx = tt.tensor(ratings['movies' if train else 'test']).to(tt.long)
        e_idx = tt.tensor(entity_ratings).to(tt.long)

        x[e_idx] = tt.tensor(1)
        y[m_idx] = tt.tensor(1)

        samples.append(x)
        labels.append(y)

    return samples, labels


if __name__ == '__main__':
    u_r_map, n_users, n_movies, n_entities = cold_start(
        from_path='../data/mindreader/user_ratings_map.json',
        conversion_map={
            -1: 1,
            0: None,  # Ignore don't know ratings
            1: 5
        },
        split_ratio=[75, 25]
    )

    u_r_map = unify_indeces(u_r_map, n_movies)

    model = NeuralNetworkAll(n_movies, n_entities, 256, 128)
    loss_fn = nn.MSELoss()
    optimizer = optimizers.Adam(model.parameters(), lr=0.001)

    n_epochs = 100
    eval_every = 1

    for e in range(n_epochs):
        xs, ys = one_hot_samples(u_r_map, n_movies, n_entities, train=True)
        for x, y in zip(xs, ys):
            model.train()

            y_hat = model(x)
            zero_map = tt.zeros_like(y)
            y_indices = y.nonzero().squeeze()
            zero_map[y_indices] = tt.tensor(1)
            loss = loss_fn(y_hat * zero_map, y.to(tt.float))
            loss.backward()
            optimizer.step()

            model.zero_grad()

        if e % eval_every == 0:
            with tt.no_grad():
                model.eval()
                train_sse = 0
                test_sse = 0
                t_xs, t_ys = one_hot_samples(u_r_map, n_movies, n_entities, train=False)
                for x, y in zip(xs, ys):
                    y_hat = model(x)
                    zero_map = tt.ones_like(y) * y.nonzero()
                    loss = (y_hat * zero_map) - y
                    train_sse += (loss ** 2).sum() / len(xs)

                for x, y in zip(t_xs, t_ys):
                    y_hat = model(x)
                    zero_map = tt.ones_like(y) * y.nonzero()
                    loss = (y_hat * zero_map) - y
                    test_sse += (loss ** 2).sum() / len(t_xs)

                print(f'Epoch {e}:')
                print(f'    Train RMSE: {tt.sqrt(train_sse)}')
                print(f'    Test RMSE:  {tt.sqrt(test_sse)}')








