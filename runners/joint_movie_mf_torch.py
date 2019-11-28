from data.training import cold_start, warm_start
from models.joint_movie_mf_torch import JointMovieMF
import torch as tt
import torch.nn as nn
import torch.optim as optimizers
import numpy as np
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


def unpack_user_ratings(u_r_map):
    train = []
    test = []

    for u, ratings in u_r_map.items():
        for m, r in ratings['movies']:
            train.append((u, m, r, 1))
        for e, r in ratings['entities']:
            train.append((u, e, r, 0))
        for m, r in ratings['test']:
            test.append((u, m, r, 1))

    shuffle(train)
    shuffle(test)

    return train, test


def sppmi(u_r_map, n_movies, n_entities):
    # Count co-occurrences between movies and entities
    n_co_occurrence_pairs = 0
    entity_rating_counts = {}
    movie_rating_counts = {}
    co_occurrence_matrix = np.zeros((n_movies, n_entities))
    for u, u_ratings in u_r_map.items():
        movies = u_ratings['movies']
        entities = u_ratings['entities']
        for m, _ in movies:
            for e, _ in entities:
                co_occurrence_matrix[m][e] += 1
                n_co_occurrence_pairs += 1

                if e not in entity_rating_counts:
                    entity_rating_counts[e] = 0
                entity_rating_counts[e] += 1

            if m not in movie_rating_counts:
                movie_rating_counts[m] = 0
            movie_rating_counts[m] += 1

    # Compute the PMI matrix
    PMI = np.zeros((n_movies, n_entities))
    for m in range(n_movies):
        for e in range(n_entities):
            if m in movie_rating_counts and e in entity_rating_counts:
                PMI[m][e] = (co_occurrence_matrix[m][e] * n_co_occurrence_pairs) \
                            / (movie_rating_counts[m] * entity_rating_counts[e])

    # Compute the SPPMI matrix
    k = 10
    SPPMI = np.zeros((n_movies, n_entities))
    for m in range(n_movies):
        for e in range(n_entities):
            SPPMI[m][e] = max(PMI[m][e] - np.log(k), 0)

    return SPPMI


def user_rates(data, n_users):
    u_likes_map = {}
    u_dislikes_map = {}

    for u in range(n_users):
        u_likes_map[u] = []
        u_dislikes_map[u] = []

    for u, m, r, is_movie in data:
        if not is_movie:
            continue

        if r == 5:
            u_likes_map[u].append(m)
        elif r == 1:
            u_dislikes_map[u].append(m)

    return u_likes_map, u_dislikes_map


def calculate_clustering_metrics(model, likes, dislikes):

    tp, tn, fp, fn = 0, 0, 0, 0

    us = tt.tensor([u for u in range(model.n_users)]).to(tt.long)
    ms = tt.tensor([m for m in range(model.n_movies)]).to(tt.long)

    for u in us:
        sim = []
        for m in ms:
            sim.append((m, model.U(u) @ model.M(m)))

        sim = sorted(sim, key=lambda x: x[1], reverse=True)

        top_movies = [i for i, s in sim[:20]]
        bottom_movies = [i for i, s in sim[-20:]]

        u_likes = likes[u.numpy().sum()]
        u_dislikes = dislikes[u.numpy().sum()]

        for m in top_movies:
            if m in u_likes:
                tp += 1
            elif m in u_dislikes:
                fp += 1

        for m in bottom_movies:
            if m in u_dislikes:
                tn += 1
            elif m in u_likes:
                fn += 1
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = (tp / (tp + fp)) if tp or fp else 0
    recall = (tn / (tn + fn)) if tn or fn else 0
    print(f'Accuracy:  {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall:    {recall}')

    print(tp, tn, fp, fn)

    return accuracy, precision, recall


if __name__ == '__main__':

    print(f'Loading data...')
    u_r_map, n_users, n_movies, n_entities = cold_start(
        from_path='../data/mindreader/user_ratings_map.json',
        conversion_map={
            -1: 1,
            0: None,  # Ignore don't know ratings
            1: 5
        },
        split_ratio=[75, 25]
    )

    print(f'Creating SPPMI matrix...')
    SPPMI = sppmi(u_r_map, n_movies, n_entities)
    train, test = unpack_user_ratings(u_r_map)

    u_likes, u_dislikes = user_rates(test, n_users)

    train = [(u, m, r, 1) for u, m, r, t in train]

    print(f'Adding SPPMI samples...')
    # Add SPPMI entries as samples
    sppmi_samples = []
    for m in range(n_movies):
        for e in range(n_entities):
            if SPPMI[m][e] > 0:
                sppmi_samples.append((e, m, SPPMI[m][e], 0))

    train += sppmi_samples
    n_epochs = 100

    print(f'Building model...')
    model = JointMovieMF(n_users, n_movies, n_entities, k=10)
    loss_fn = nn.MSELoss()
    optimizer = optimizers.Adam(model.parameters(), lr=0.001)

    train_history = []
    test_history = []

    accuracy_history = []
    precision_history = []
    recall_history = []



    print(f'Beginning training...')
    for e in range(n_epochs):
        shuffle(train)
        print(f'Beginning epoch {e}...')

        for users, movies, ratings, is_users in batches(train, n=64):
            model.train()

            y_hat = model(users, movies, is_users)
            loss = loss_fn(y_hat, ratings)

            loss.backward()
            optimizer.step()

            model.zero_grad()

        if e % 10 == 0:
            with tt.no_grad():
                print(f'Evaluating...')
                model.eval()
                train_users, train_movies, train_ratings, train_is_user = to_tensors(*zip(*train))
                test_users, test_movies, test_ratings, test_is_user = to_tensors(*zip(*test))

                train_y_hat = model(train_users, train_movies, train_is_user)
                test_y_hat = model(test_users, test_movies, test_is_user)

                train_loss = loss_fn(train_y_hat, train_ratings)
                test_loss = loss_fn(test_y_hat, test_ratings)

                train_history.append(train_loss)
                test_history.append(test_loss)

                acc, prec, rec = calculate_clustering_metrics(model, u_likes, u_dislikes)
                accuracy_history.append(acc)
                precision_history.append(prec)
                recall_history.append(rec)

                print(f'Epoch {e}:')
                print(f'    Train RMSE: {tt.sqrt(train_loss)}')
                print(f'    Test RMSE:  {tt.sqrt(test_loss)}')

    plt.plot(accuracy_history, color='orange', label='Accuracy')
    plt.plot(precision_history, color='skyblue', label='Precision')
    plt.plot(recall_history, color='green', label='Recall')
    plt.legend()
    plt.title(f'Joint Movie Embedding ({n_epochs} epochs) - top/bottom 20, movies and entities')
    plt.show()

    plt.plot(train_history, color='orange', label='Train RMSE')
    plt.plot(test_history, color='skyblue', label='Test RMSE')
    plt.legend()
    plt.title(f'Joint Movie Embedding ({n_epochs} epochs) - top/bottom 20, movies and entities')
    plt.show()



