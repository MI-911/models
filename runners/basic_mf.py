from random import shuffle

import torch
from torch import nn, tensor

from data.training import warm_start
from models.mf import MF, MFExtended
import torch.optim as optimizers

from utilities.util import combine_movie_entity_index


def batch_generator(data, batch_size):
    length = len(data)
    step = 0
    while True:
        cur_index = batch_size * step
        batch = data[cur_index: cur_index + batch_size]
        step += 1

        batch = list(zip(*batch))
        batch = torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.FloatTensor(batch[2])

        if cur_index + batch_size < length:
            yield batch
        else:
            return batch


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

    c = combine_movie_entity_index(train + test)

    n_iter = 30000
    k = 20
    lr = 1e-4
    eval_every = 10

    model = MFExtended(n_users, n_movies+n_entities, k)
    optimizer = optimizers.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train = [(u, c[type][m], rating) for u, m, rating, type in train]
    test = [(u, c[type][m], rating) for u, m, rating, type in test]

    for n in range(n_iter):
        train_generator = batch_generator(train, 264)
        test_generator = batch_generator(test, 264)

        step = 0
        tot_loss = 0
        for users, entities, ratings in train_generator:
            step += 1
            prediction = model(users, entities)
            loss = loss_fn(prediction, ratings)
            tot_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if step != 0 and n % 25 == 0:
            print(f'Iter: {n}')
            print(f'Loss: {tot_loss / step}')
        shuffle(train)

        with torch.no_grad():
            step = 0
            tot_loss = 0
            for users, entities, ratings in test_generator:
                step += 1
                prediction = model(users, entities)
                loss = loss_fn(prediction, ratings)
                tot_loss += loss
            if n % 25 == 0:
                print(f'Test Loss: {tot_loss / step}')



