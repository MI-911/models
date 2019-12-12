import numpy as np
import pandas as pd
import json
import torch as tt
import torch.optim as optim
import random
from models.trans_e import TransE
import matplotlib.pyplot as plt


MINDREADER_GRAPH_TRIPLES_PATH = '../data/graph/triples.csv'


def split(lst, ratio):
    split_idx = int(len(lst) * ratio)
    return np.array(lst[:split_idx], dtype=np.int), np.array(lst[split_idx:], dtype=np.int)


def adjacency_map(triples):
    adj_map = {}
    for h, r, t in triples:
        if h not in adj_map:
            adj_map[h] = {}
        if r not in adj_map[h]:
            adj_map[h][r] = []
        if t not in adj_map[h][r]:
            adj_map[h][r].append(t)

    return adj_map


def indexize_triples(path):
    with open(path) as fp:
        df = pd.read_csv(fp)

    triples = [(head, relation, tail) for head, relation, tail in df[['head_uri', 'relation', 'tail_uri']].values]

    with open('../data/mindreader/ratings_clean.json') as fp:
        ratings = json.load(fp)
        ratings = [(u, 'likes' if r == 1 else 'dislikes', e) for u, e, r in ratings if r == 1 or r == -1]
        triples += ratings

    # Convert URIs to indices
    entity_index_map, ec = {}, 0
    relation_index_map, rc = {}, 0

    for h, r, t in triples:
        if h not in entity_index_map:
            entity_index_map[h] = ec
            ec += 1
        if t not in entity_index_map:
            entity_index_map[t] = ec
            ec += 1
        if r not in relation_index_map:
            relation_index_map[r] = rc
            rc += 1

    # Convert triples to index-form
    indexed_triples = [(entity_index_map[h], relation_index_map[r], entity_index_map[t]) for h, r, t in triples]

    return indexed_triples, ec, rc


def corrupt(h, r, t):
    if random.random() > 0.5:
        # Corrupt the head
        return random.choice(all_entities), r, t
    else:
        # Corrupt the tail
        return h, r, random.choice(all_entities)


def corrupt_batch(hs, rs, ts):
    corrupted_triples = []
    for h, r, t in zip(hs, rs, ts):
        corrupted_triples.append(corrupt(h, r, t))

    return training_batch(corrupted_triples)


def generate_batches(iterable, batch_size=64):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield training_batch(iterable[ndx:min(ndx + batch_size, l)])


def training_batch(triples):
    heads, relations, tails = zip(*triples)
    return tt.tensor(heads), tt.tensor(relations), tt.tensor(tails)


def evaluate_precision(model, triples, n=10):
    mean_ranks = []
    for i, (h, r, t) in enumerate(triples):
        mean_ranks.append(model.fast_validate(h, r, t))
        # if i % 1000 == 0:
        #     print(f'Evaluating ({(i / len(triples)) * 100 : 2.2f}%)')

    print(np.mean(mean_ranks))
    mean_ranks_history.append(np.mean(mean_ranks))


def split_training_test_set(triples):
    es, rs = set(), set()

    train = set()
    test = set()

    n_triples = len(triples)
    n_training_triples = int(n_triples * 0.85)

    print(f'Extracting {n_training_triples} ({(n_training_triples / n_triples) * 100 : 2.2f}%) training triples from {n_triples} triples')

    for h, r, t in triples:
        if h not in es or t not in es:
            es.add(h)
            es.add(t)
            train.add((h, r, t))
        else:
            test.add((h, r, t))

    remaining_in_train = n_training_triples - len(train)
    print(f'Currently have {len(train)} training triples, need {n_training_triples}. Picking {remaining_in_train} triples from {len(test)} test triples.')

    test = list(test)
    random.shuffle(test)
    to_add_to_training = set(test[:remaining_in_train])

    print(f'Picked {len(to_add_to_training)} triples from test.')
    train = train.union(to_add_to_training)
    test = set(test) - to_add_to_training

    print(f'We now have {len(train)} training triples and {len(test)} testing triples')

    print(f'All (h, r, ?) and (?, r, t) pairs in test are also in training')
    print(f'We have {len(train)} ({(len(train) / n_triples) * 100 : 2.2f}%) training triples and {len(test)} ({(len(test) / n_triples) * 100 : 2.2f})% testing triples.')
    return list(train), list(test)


if __name__ == '__main__':

    # Load data
    triples, n_entities, n_relations = indexize_triples(MINDREADER_GRAPH_TRIPLES_PATH)
    random.shuffle(triples)
    train_triples, test_triples = split_training_test_set(triples)
    # train_triples, test_triples = split(triples, 0.75)

    random.shuffle(train_triples)
    random.shuffle(test_triples)

    entities_in_train = set([h for h, r, t in train_triples])
    tails_in_train = set([t for h, r, t in train_triples])
    relations_in_train = set([r for h, r, t in train_triples])
    entities_in_test = set([h for h, r, t in test_triples])
    tails_in_test = set([t for h, r, t in test_triples])
    relations_in_test = set([r for h, r, t in test_triples])

    print(f'{len(entities_in_train)} heads, {len(relations_in_train)} relations and {len(tails_in_train)} tails in training')
    print(f'{len(entities_in_test)} heads, {len(relations_in_test)} relations and {len(tails_in_test)} tails in test')

    print(f'{len(entities_in_train - entities_in_test)} heads, {len(relations_in_train - relations_in_test)} relations and {len(tails_in_train - tails_in_test)} tails in train not in test')
    print(f'{len(entities_in_test - entities_in_train)} heads, {len(relations_in_test - relations_in_train)} relations and {len(tails_in_test - tails_in_train)} tails in test not in train')

    # For the corrupt function
    all_entities = [i for i in range(n_entities)]

    # Initialize the TransE learning model
    k = 10
    margin = 1.0
    trans_e = TransE(n_entities=n_entities, n_relations=n_relations, margin=margin, k=k)
    optimizer = optim.SGD(trans_e.parameters(), lr=0.003)
    loss_fn = tt.nn.MarginRankingLoss(trans_e.margin).to(trans_e.device)

    # Begin training loop
    n_training_triples = len(train_triples)
    n_epochs = 50
    batch_size = 24

    training_loss_history = []
    mean_ranks_history = []

    def evaluate(triples, loss_str):
        with tt.no_grad():
            trans_e.eval()

            h, r, t = training_batch(triples)
            c_h, c_r, c_t = corrupt_batch(h, r, t)

            p_loss = trans_e(h, r, t)
            n_loss = trans_e(c_h, c_r, c_t)

            tmp_tensor = tt.tensor([-1], dtype=tt.float).to(trans_e.device)
            batch_loss = tt.cat((p_loss, n_loss))
            p_loss = batch_loss.view(2, -1)[0]
            n_loss = batch_loss.view(2, -1)[1]
            batch_loss = loss_fn(p_loss, n_loss, tmp_tensor)

            print(f'Epoch {epoch}: {loss_str}: {batch_loss}')
            return batch_loss

    for epoch in range(n_epochs):

        training_loss_history.append(evaluate(train_triples, 'Training loss'))
        test_loss = evaluate(test_triples, 'Testing loss')

        random.shuffle(train_triples)
        training_batches = generate_batches(train_triples)

        for h, r, t in training_batches:
            trans_e.train()
            c_h, c_r, c_t = corrupt_batch(h, r, t)
            p_loss = trans_e(h, r, t)
            n_loss = trans_e(c_h, c_r, c_t)

            tmp_tensor = tt.tensor([-1], dtype=tt.float).to(trans_e.device)
            batch_loss = tt.cat((p_loss, n_loss))
            p_loss = batch_loss.view(2, -1)[0]
            n_loss = batch_loss.view(2, -1)[1]
            batch_loss = loss_fn(p_loss, n_loss, tmp_tensor)
            batch_loss.backward()
            optimizer.step()

            trans_e.zero_grad()

    plt.plot(training_loss_history)
    plt.title('TransE training')
    plt.xlabel('Epochs')
    plt.ylabel('Mean distance between e(h) + e(r) and e(t) for some (h, r, t)')
    plt.show()

    plt.plot(mean_ranks_history)
    plt.title('TransE training')
    plt.xlabel('Epochs')
    plt.ylabel('Mean ranking of head/tail entities for all (h,r,t) triples')
    plt.show()