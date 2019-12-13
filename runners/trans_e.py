import numpy as np
import pandas as pd
import json
import torch as tt
import torch.optim as optim
import random
from models.trans_e import TransE
import matplotlib.pyplot as plt


MINDREADER_GRAPH_TRIPLES_PATH = '../data/graph/triples.csv'
MINDREADER_RATINGS_BASE_PATH = '../data/mindreader'


class User:
    def __init__(self, index):
        self.index = index
        self.liked_movies = []
        self.disliked_movies = []
        self.liked_entities = []
        self.disliked_entities = []


def split(lst, ratio):
    split_idx = int(len(lst) * ratio)
    return lst[:split_idx], lst[split_idx:]


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


def create_users(path, n_entities):
    users = {}
    with open(f'{path}/ratings_clean.json') as rp:
        ratings = json.load(rp)
    with open(f'{path}/entities_clean.json') as ep:
        entities = json.load(ep)

    # Map URIs to labels
    label_map = {}
    for uri, name, labels in entities:
        labels = labels.split('|')
        if uri not in label_map:
            label_map[uri] = labels

    # Filter out unknowns?
    ratings = [(u, e, r) for u, e, r in ratings if not r == 0]

    movie_ratings = [(u, m, r) for u, m, r in ratings if 'Movie' in label_map[m]]
    entity_ratings = [(u, m, r) for u, m, r in ratings if 'Movie' not in label_map[m]]

    entity_idx_map, ec = {}, n_entities
    for u, m, r in movie_ratings + entity_ratings:
        if u not in entity_idx_map:
            entity_idx_map[u] = ec
            ec += 1
        if m not in entity_idx_map:
            entity_idx_map[m] = ec
            ec += 1

    movie_ratings = [(entity_idx_map[u], entity_idx_map[m], r) for u, m, r in movie_ratings]
    entity_ratings = [(entity_idx_map[u], entity_idx_map[m], r) for u, m, r in entity_ratings]

    users = {}
    for u, m, r in movie_ratings:
        if u not in users:
            users[u] = User(u)
        if r == 1:
            users[u].liked_movies.append(m)
        else:
            users[u].disliked_movies.append(m)
    for u, m, r in entity_ratings:
        if u not in users:
            users[u] = User(u)
        if r == 1:
            users[u].liked_entities.append(m)
        else:
            users[u].disliked_entities.append(m)

    return users, ec


def indexize_ratings(path, n_entities, n_relations):
    # Starts indexing entities and movies from n_entities and relations from n_relations and onwards

    with open(f'{path}/ratings_clean.json') as rp:
        ratings = json.load(rp)
    with open(f'{path}/entities_clean.json') as ep:
        entities = json.load(ep)

    # Map URIs to labels
    label_map = {}
    for uri, name, labels in entities:
        labels = labels.split('|')
        if uri not in label_map:
            label_map[uri] = labels

    # Filter out unknowns?
    ratings = [(u, e, r) for u, e, r in ratings if not r == 0]

    movie_ratings = [(u, m, r) for u, m, r in ratings if 'Movie' in label_map[m]]
    entity_ratings = [(u, m, r) for u, m, r in ratings if 'Movie' not in label_map[m]]

    entity_idx_map, ec = {}, n_entities
    movie_indices = []
    user_indices = []

    for u, m, r in movie_ratings:
        if u not in entity_idx_map:
            entity_idx_map[u] = ec
            user_indices.append(ec)
            ec += 1
        if m not in entity_idx_map:
            entity_idx_map[m] = ec
            movie_indices.append(ec)
            ec += 1
    for u, e, r in entity_ratings:
        if u not in entity_idx_map:
            entity_idx_map[u] = ec
            ec += 1
        if e not in entity_idx_map:
            entity_idx_map[e] = ec
            ec += 1

    # Create two/three new relations
    rc = n_relations
    like_relation_index = rc
    rc += 1
    dislike_relation_index = rc
    rc += 1
    unknown_relation_index = rc
    rc += 1

    triples = [(entity_idx_map[u], like_relation_index if r == 1 else dislike_relation_index, entity_idx_map[e]) for u, e, r in ratings]
    return triples, ec, rc, user_indices, movie_indices, like_relation_index


def indexize_triples(path):
    with open(path) as fp:
        df = pd.read_csv(fp)

    triples = [(head, relation, tail) for head, relation, tail in df[['head_uri', 'relation', 'tail_uri']].values]

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


def evaluate_rank(model, triples):
    mean_ranks = []
    for i, (h, r, t) in enumerate(triples):
        mean_ranks.append(model.fast_validate(h, r, t))
        # if i % 1000 == 0:
        #     print(f'Evaluating ({(i / len(triples)) * 100 : 2.2f}%)')

    mean_rank = np.mean(mean_ranks)
    mean_ranks_history.append(mean_rank)
    return mean_rank


def evaluate_precision(model, matrix, user_indices, movie_indices, n=20):
    u_revert = {m: i for i, m in u_idx_matrix_map.items()}
    m_revert = {m: i for i, m in m_idx_matrix_map.items()}

    average_precisions = []

    u_matrix_indices = np.where(matrix.sum(axis=1) > 0)[0]

    uc = 0
    for u in u_matrix_indices:
        ratings_vector = matrix[u]
        u_idx = u_revert[u]
        liked_movies = np.where(ratings_vector > 0)[0]
        liked_movies = [m_revert[m] for m in liked_movies]

        ranked_movies = model.predict_movies_for_user(u_idx, like_relation_index, movie_indices)
        ranked_movies = sorted(ranked_movies, key=lambda x: x[1], reverse=True)

        precisions = []
        for i in range(1, n + 1, 1):
            tp = 0
            fp = 0
            for k in range(i):
                if ranked_movies[i] in liked_movies:
                    tp += 1
                else:
                    fp += 1
            precisions.append(tp / (tp + fp))

        average_precisions.append(np.mean(precisions))

        uc += 1

        if uc % 100 == 0:
            print(f'Calculating precision ({(uc / len(u_matrix_indices)) * 100 : 2.2f}%)')

    return np.mean(average_precisions)


def balanced_split(triples):
    adj_map = adjacency_map(triples)

    train, test = [], []

    for h, rs in adj_map.items():
        for r, ts in rs.items():
            random.shuffle(ts)
            tr, te = split(ts, 0.8)
            train += [(h, r, t) for t in tr]
            test += [(h, r, t) for t in te]

    return train, test


def reverse_triples(triples, num_relations):
    complement_relation_map = {}
    reversed_triples = []
    for h, r, t in triples:
        if r not in complement_relation_map:
            complement_relation_map[r] = num_relations
            num_relations += 1

        reversed_triples.append((t, complement_relation_map[r], h))

    return triples + reversed_triples, num_relations


def split_training_test_set(triples):
    es, rs = set(), set()

    train = set()
    test = set()

    n_triples = len(triples)
    n_training_triples = int(n_triples * 0.85)

    print(f'Extracting {n_training_triples} ({(n_training_triples / n_triples) * 100 : 2.2f}%) training triples from {n_triples} triples')

    for h, r, t in triples:
        if h not in es:
            es.add(h)
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
    print(f'We have {len(train)} ({(len(train) / n_triples) * 100 : 2.2f}%) training triples and {len(test)} ({(len(test) / n_triples) * 100 : 2.2f}%) testing triples.')
    return list(train), list(test)


def get_user_movie_index_map(triples, user_indices, movie_indices):

    u_idx_matrix_map, uc = {}, 0
    m_idx_matrix_map, mc = {}, 0

    for h, r, t in triples:
        if h in user_indices and t in movie_indices:
            if h not in u_idx_matrix_map:
                u_idx_matrix_map[h] = uc
                uc += 1
            if t not in m_idx_matrix_map:
                m_idx_matrix_map[t] = mc
                mc += 1

    return u_idx_matrix_map, m_idx_matrix_map


if __name__ == '__main__':

    # Load data
    triples, n_entities, n_relations = indexize_triples(MINDREADER_GRAPH_TRIPLES_PATH)

    rating_triples, n_entities, n_relations, user_indices, movie_indices, like_relation_index = indexize_ratings(MINDREADER_RATINGS_BASE_PATH, n_entities, n_relations)

    triples += rating_triples
    random.shuffle(triples)

    triples, n_relations = reverse_triples(triples, n_relations)
    # train_triples, test_triples = balanced_split(triples)
    # train_triples, test_triples = split_training_test_set(triples)

    train_triples, test_triples = split(triples, 0.75)

    # Create ratings matrices for user/movie pairs for later evaluation
    u_idx_matrix_map, m_idx_matrix_map = get_user_movie_index_map(triples, user_indices, movie_indices)
    train_ratings_matrix = np.zeros((len(user_indices), len(movie_indices)))
    test_ratings_matrix = np.zeros((len(user_indices), len(movie_indices)))
    for h, r, t in train_triples:
        if h in user_indices and t in movie_indices:
            u = u_idx_matrix_map[h]
            m = m_idx_matrix_map[t]
            train_ratings_matrix[u][m] = 1 if r == like_relation_index else -1
    for h, r, t in test_triples:
        if h in user_indices and t in movie_indices:
            u = u_idx_matrix_map[h]
            m = m_idx_matrix_map[t]
            test_ratings_matrix[u][m] = 1 if r == like_relation_index else -1

    random.shuffle(train_triples)
    random.shuffle(test_triples)

    heads_in_train = set([h for h, r, t in train_triples])
    tails_in_train = set([t for h, r, t in train_triples])
    relations_in_train = set([r for h, r, t in train_triples])
    heads_in_test = set([h for h, r, t in test_triples])
    tails_in_test = set([t for h, r, t in test_triples])
    relations_in_test = set([r for h, r, t in test_triples])

    print(f'{len(heads_in_train) + len(tails_in_train)} entities in train')
    print(f'{len(heads_in_test) + len(tails_in_test)} entities in test')
    print(f'{len(heads_in_train - heads_in_test) + len(tails_in_train - tails_in_test)} entities in train not in test')
    print(f'{len(heads_in_test - heads_in_train) + len(tails_in_test - tails_in_train)} entities in test not in train')

    print(f'In training, {len(heads_in_train - tails_in_train)} entities appear as heads but not as tails')
    print(f'In training, {len(tails_in_train - heads_in_train)} entities appear as tails but not as heads')
    print(f'In testing, {len(tails_in_test - heads_in_test)} entities appear as tailsSGD but not as heads')
    print(f'In testing, {len(tails_in_test - heads_in_test)} entities appear as tails but not as heads')

    print(f'{len(heads_in_train)} heads, {len(relations_in_train)} relations and {len(tails_in_train)} tails in training')
    print(f'{len(heads_in_test)} heads, {len(relations_in_test)} relations and {len(tails_in_test)} tails in test')

    print(f'{len(heads_in_train - heads_in_test)} heads, {len(relations_in_train - relations_in_test)} relations and {len(tails_in_train - tails_in_test)} tails in train not in test')
    print(f'{len(heads_in_test - heads_in_train)} heads, {len(relations_in_test - relations_in_train)} relations and {len(tails_in_test - tails_in_train)} tails in test not in train')

    # For the corrupt function
    all_entities = [i for i in range(n_entities)]

    # Initialize the TransE learning model
    k = 100
    margin = 1.0
    trans_e = TransE(n_entities=n_entities, n_relations=n_relations, margin=margin, k=k)
    optimizer = optim.Adam(trans_e.parameters(), lr=0.003)
    loss_fn = tt.nn.MarginRankingLoss(trans_e.margin).to(trans_e.device)

    # Begin training loop
    n_training_triples = len(train_triples)
    n_epochs = 250
    batch_size = 64

    training_loss_history = []
    mean_ranks_history = []

    def evaluate(triples, loss_str):
        with tt.no_grad():
            trans_e.eval()

            mean_rank = None

            if epoch % 5 == 0 and epoch > 0:
                # mean_rank = evaluate_rank(trans_e, triples)
                print(f'{loss_str} mean rank: {mean_rank}')

            h, r, t = training_batch(triples)
            c_h, c_r, c_t = corrupt_batch(h, r, t)

            p_loss = trans_e(h, r, t)
            n_loss = trans_e(c_h, c_r, c_t)

            tmp_tensor = tt.tensor([-1], dtype=tt.float).to(trans_e.device)
            batch_loss = tt.cat((p_loss, n_loss))
            p_loss = batch_loss.view(2, -1)[0]
            n_loss = batch_loss.view(2, -1)[1]
            batch_loss = loss_fn(p_loss, n_loss, tmp_tensor)

            return batch_loss

    for epoch in range(n_epochs):

        print(f'Beginning epoch {epoch}...')

        training_loss = evaluate(train_triples, 'Training')
        test_loss = evaluate(test_triples, 'Testing')

        # training_map = evaluate_precision(trans_e, train_ratings_matrix, user_indices, movie_indices, n=20)
        # testing_map = evaluate_precision(trans_e, test_ratings_matrix, user_indices, movie_indices, n=20)

        print(f'Epoch {epoch}:')
        print(f'    Train: loss: {training_loss}')
        print(f'    Test:  loss: {test_loss}')

        training_loss_history.append(training_loss)

        random.shuffle(train_triples)
        training_batches = generate_batches(train_triples)

        for h, r, t in training_batches:
            trans_e.train()
            c_h, c_r, c_t = corrupt_batch(h, r, t)
            p_loss = trans_e(h, r, t)
            n_loss = trans_e(c_h, c_r, c_t)

            tmp_tensor = tt.tensor([-1], dtype=tt.float).to(trans_e.device)
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


# TODO: 1. Make sure that standard (h, r, t) triples can be differentiated from (user, likes/dislikes, movie/entity)
#          triples.
#       2. Make sure that user rating triples are corrupted in the correct way, and not like other triples.
#       3. Calculate Precision@20, AP@20 both for user-rates-movie and user-rates-entity triples.
#       4. Use the ranking metric to show if corrupting user-rates-entity triplets in the designed way improves
#          performance. If ranking doesn't help, look at Precision@20, AP@20, etc.
