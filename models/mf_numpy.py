from data.training import warm_start
import numpy as np
from random import shuffle
from matplotlib import pyplot as plt


def euclidean_distance(v, u):
    ss = 0
    for e1, e2 in zip(v, u):
        ss += (e1 - e2) ** 2

    return np.sqrt(ss)


def calculate_clustering_metrics(U, I, likes, dislikes):

    tp, tn, fp, fn = 0, 0, 0, 0

    all_sims = []

    for u, uv in enumerate(U):
        sim = []
        for i, iv in enumerate(I):
            sim.append((i, uv @ iv))

        sim = sorted(sim, key=lambda x: x[1], reverse=True)

        top_movies = [i for i, s in sim[:20]]
        bottom_movies = [i for i, s in sim[-20:]]

        u_likes = likes[u]
        u_dislikes = dislikes[u]

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


def calculate_rmse(data, U, M):
    # Compute the predicted rating matrix
    predicted = U @ M.T

    n = len(data)
    sse = 0

    for u, m, r, is_movie in data:
        sse += (predicted[u][m] - r) ** 2

    return np.sqrt(sse / n)


if __name__ == '__main__':

    n_latent_factors = 25
    learning_rate = 0.001
    regularize = 0.001
    n_iter = 200

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

    u_likes, u_dislikes = user_rates(test, n_users)

    U = np.random.rand(n_users, n_latent_factors)
    M = np.random.rand(n_movies, n_latent_factors)

    training_history = []
    testing_history = []

    accuracy_history = []
    precision_history = []
    recall_history = []

    train = [(u, m if is_movie else m + n_movies, r, is_movie) for u, m, r, is_movie in train if is_movie]

    # Filter out movies?
    train = [(u, m, r, is_movie) for u, m, r, is_movie in train if is_movie]

    for i in range(n_iter):
        shuffle(train)
        train_rmse = calculate_rmse(train, U, M)
        test_rmse = calculate_rmse(test, U, M)

        training_history.append(train_rmse)
        testing_history.append(test_rmse)

        for u, m, r, is_movie in train:
            error = U[u] @ M[m] - r

            for k in range(n_latent_factors):
                u_gradient = error * M[m][k]
                U[u][k] -= learning_rate * (u_gradient - regularize * M[m][k])

                m_gradient = error * U[u][k]
                M[m][k] -= learning_rate * (m_gradient - regularize * U[u][k])

        print(f'Iteration {i}: Train RMSE: {train_rmse} (Test RMSE: {test_rmse})')
        if i % 10 == 0:
            acc, prec, rec = calculate_clustering_metrics(U, M[list(range(n_movies))], u_likes, u_dislikes)

            accuracy_history.append(acc)
            precision_history.append(prec)
            recall_history.append(rec)

    plt.plot(accuracy_history, color='skyblue', label='Accuracy')
    plt.plot(precision_history, color='orange', label='Precision')
    plt.plot(recall_history, color='green', label='Recall')
    plt.plot()
    plt.legend()
    plt.title('Basic MF (top/bottom 20) (train on movies only)')
    plt.show()

    plt.plot(training_history, color='blue', label='Train RMSE')
    plt.plot(testing_history, color='red', label='Test RMSE')
    plt.plot()
    plt.legend()
    plt.title('Basic MF (top/bottom 20) (train on movies only)')
    plt.show()


