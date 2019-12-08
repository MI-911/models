from random import shuffle

import torch
from torch import nn, FloatTensor, LongTensor
import torch.optim as optimizers
from tqdm import tqdm

from data.adjancency_matrix_loader import load_adjacency_matrix
from data.training import warm_start
from models.label_smoothness import GNN



if __name__ == '__main__':
    train, test, n_users, n_movies, n_descriptive_entities = warm_start(
        ratings_path='../data/mindreader/ratings_clean.json',
        entities_path='../data/mindreader/entities_clean.json',
        conversion_map={
            -1: 1,
            0: None,  # Ignore don't know ratings
            1: 5
        },
        split_ratio=[75, 25],
        create_uri_indices=False
    )

    values, indices, entity_index, relation_index, n_entities = load_adjacency_matrix('../data/mindreader/triples.csv')

    # Convert to adjacency matrix indices.
    train = [(LongTensor([userId]), LongTensor([entity_index[uri]]), FloatTensor([rating]))
             for (userId, uri, rating, _) in train]
    test = [(LongTensor([userId]), LongTensor([entity_index[uri]]), FloatTensor([rating]))
            for (userId, uri, rating, _) in test]

    n_iter = 30000
    k = 20
    lr = 1e-4
    eval_every = 10

    model = GNN(n_users, len(relation_index), n_entities, values, indices, )
    optimizer = optimizers.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for n in range(n_iter):
        step = 0
        tot_loss = 0
        for users, entities, ratings in tqdm(train):
            step += 1
            prediction = model(users, entities)
            loss = loss_fn(prediction, ratings)
            tot_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if step != 0 and n % 1 == 0:
            print(f'Iter: {n}')
            print(f'Loss: {tot_loss / step}')
        shuffle(train)

        with torch.no_grad():
            step = 0
            tot_loss = 0
            for users, entities, ratings in train:
                step += 1
                prediction = model(users, entities)
                loss = loss_fn(prediction, ratings)
                tot_loss += loss
            if n % 1 == 0:
                print(f'Test Loss: {tot_loss / step}')

    print('a')

