from random import shuffle

import torch
from torch import nn, FloatTensor, LongTensor
import torch.optim as optimizers
from torch.nn import functional as F
from tqdm import tqdm

from data.adjancency_matrix_loader import load_adjacency_matrix
from data.training import warm_start
from models.label_smoothness import GNN
from utilities.util import batch_generator
import os


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"


    train, test, n_users, n_movies, n_descriptive_entities = warm_start(
        ratings_path='../data/mindreader/ratings_clean.json',
        entities_path='../data/mindreader/entities_clean.json',
        conversion_map={
            -1: 0,
            0: None,  # Ignore don't know ratings
            1: 1
        },
        split_ratio=[75, 25],
        create_uri_indices=False
    )

    values, indices, entity_index, relation_index, n_entities = load_adjacency_matrix('../data/mindreader/triples.csv')

    # Convert to adjacency matrix indices.
    train = [(LongTensor([userId]), LongTensor([entity_index[uri]]), LongTensor([rating]))
             for (userId, uri, rating, _) in train]
    test = [(LongTensor([userId]), LongTensor([entity_index[uri]]), LongTensor([rating]))
            for (userId, uri, rating, _) in test]

    n_iter = 30000
    k = 20
    lr = 1e-4
    eval_every = 10
    batch_size = 4

    model = GNN(n_users, len(relation_index), n_entities, values, indices, batch_size=batch_size).cuda()
    optimizer = optimizers.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for n in range(n_iter):
        train_gen = batch_generator(train, batch_size, True)
        train_tqdm = tqdm(train_gen, total=(len(train)//batch_size), position=0)
        test_gen = batch_generator(test, batch_size, True)
        test_tqdm = tqdm(test_gen, total=(len(test)//batch_size), position=1)

        step = 0
        last_10_loss = [0]*10
        last_loss_i = 0
        for users, entities, ratings in train_tqdm:
            step += 1
            prediction = model(users, entities)
            loss = loss_fn(prediction, ratings)
            last_10_loss[last_loss_i] = loss
            last_loss_i = (last_loss_i+1) % 10

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_tqdm.set_description('ML (loss=%.2f)' % (sum(last_10_loss) / 10))

        shuffle(train)

        with torch.no_grad():
            step = 0
            tot_loss = 0

            last_10_loss = [0] * 10
            last_loss_i = 0
            for users, entities, ratings in test_tqdm:
                step += 1
                prediction = model(users, entities)
                loss = loss_fn(prediction, ratings)
                tot_loss += loss
                last_10_loss[last_loss_i] = loss
                last_loss_i = (last_loss_i+1) % 10
                test_tqdm.set_description('ML (loss=%g)' % (sum(last_10_loss) / 10))

            test_tqdm.set_description('ML (loss=%g)' % (tot_loss / step))

    print('a')

