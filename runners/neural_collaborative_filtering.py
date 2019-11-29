from random import shuffle

import torch
from torch import nn

from data.training import warm_start
from models.neural_collaborative_filtering import NCF
from utilities.util import combine_movie_entity_index, batch_generator

import torch.optim as optimizers


if __name__ == '__main__':
    train, test, n_users, n_movies, n_entities = warm_start(
        ratings_path='../data/mindreader/ratings_clean.json',
        entities_path='../data/mindreader/entities_clean.json',
        conversion_map={
            -1: 0,
            0: None,  # Ignore don't know ratings
            1: 1
        },
        split_ratio=[75, 25]
    )

    c = combine_movie_entity_index(train + test)

    n_iter = 30000
    k = 8
    lr = 1e-5
    eval_every = 10

    model = NCF(n_users, n_movies+n_entities, k)
    optimizer = optimizers.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss(reduction='sum')

    train = [(u, c[type][m], rating) for u, m, rating, type in train]
    test = [(u, c[type][m], rating) for u, m, rating, type in test]

    for n in range(n_iter):
        train_generator = batch_generator(train, 8)
        test_generator = batch_generator(test, 8)

        step = 0
        tot_loss = 0
        for users, entities, ratings in train_generator:
            users = nn.functional.one_hot(users, n_users)
            entities = nn.functional.one_hot(entities, n_movies + n_entities)
            users = users.float()
            entities = entities.float()

            step += 1
            prediction = model(users, entities)
            loss = loss_fn(prediction, ratings)
            tot_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if step != 0 and n % 2 == 0:
            print(f'Iter: {n}')
            print(f'Loss: {tot_loss / step}')
        shuffle(train)

        with torch.no_grad():
            step = 0
            tot_loss = 0
            tot = 0
            correct = 0
            for users, entities, ratings in test_generator:
                users = nn.functional.one_hot(users, n_users)
                entities = nn.functional.one_hot(entities, n_movies + n_entities)
                users = users.float()
                entities = entities.float()

                step += 1
                prediction = model(users, entities)
                loss = loss_fn(prediction, ratings)
                tot_loss += loss

                if n % 2 == 0:
                    labels = [1 if pred >= 0.5 else 0 for pred in prediction]
                    for true, pred in zip(ratings, labels):
                        tot += 1
                        if true == pred:
                            correct += 1

            if n % 2 == 0:
                print(f'Test Loss: {tot_loss / step} \t ACC: {correct / tot}')