import numpy as np
from numpy.linalg import inv, solve
import pandas as pd
import random
from random import shuffle
import json


def load_movielens_ratings():
    with open(f'../data/movielens/ratings.csv') as fp:
        df = pd.read_csv(fp)
        ratings = [(u, m, r) for u, m, r in df[['userId', 'movieId', 'rating']].values]

    # Map the ids to indices
    u_map, uc = {}, 0
    m_map, mc = {}, 0

    for u, m, r in ratings:
        if u not in u_map:
            u_map[u] = uc
            uc += 1
        if m not in m_map:
            m_map[m] = mc
            mc += 1

    # Convert to integers
    ratings = [(u_map[u], m_map[m], r) for u, m, r in ratings]

    split_index = int(len(ratings) * 0.75)

    shuffle(ratings)

    train = ratings[:split_index]
    test = ratings[split_index:]
    return train, test, uc, mc


def user_major_ratings(ratings):
    user_ratings = {}
    for u, m, r in ratings:
        if u not in user_ratings:
            user_ratings[u] = []
        user_ratings[u].append((m, r))

    for u, ratings in list(user_ratings.items()):
        user_ratings[u] = {m: r for m, r in ratings}

    return user_ratings


def ratings_matrix(ratings, n_users, n_items):
    M = np.zeros((n_users, n_items))
    for u, m, r in ratings: 
        M[u][m] = r 
        
    return M


def co_occurrence_sppmi_v2(ratings_matrix):
    n_users, n_items = ratings_matrix.shape
    one_hot_ratings_matrix = ratings_matrix / ratings_matrix
    co_occurrence_matrix = np.zeros((n_items, n_items))

    for i in range(n_items):
        pass


def co_occurrence_sppmi(ratings, n_users, n_items):
    # hash(i,j) = number of users who consumed both item i and j
    # hash(i) = sum of number of users who consumed both item i and j for every j
    # hash(j) = sum of number of users who consumed both item i and j for every i
    # D = number of users? May be the total number of co-occurrence pairs? Number of ratings?

    # Calculate per user ratings
    user_ratings = user_major_ratings(ratings)

    n_ratings = len(ratings)

    # Define the hashij matrix
    n_co_occurences = np.zeros((n_items, n_items))
    for u, ratings in user_ratings.items():
        rated_items = list(ratings.keys())
        for _i, i in enumerate(rated_items):
            for j in rated_items[_i + 1:]:
                n_co_occurences[i][j] += 1
                n_co_occurences[j][i] += 1

    # Calculate the PMI
    pointwise_mutual_information = (
        (n_co_occurences * n_ratings) /
        np.clip((n_co_occurences.sum(axis=1).reshape((n_items, 1)) @ n_co_occurences.sum(axis=0).reshape((1, n_items))),
                1, np.inf)
    )

    #
    # pointwise_mutual_information = np.zeros((n_items, n_items))
    # for i in range(n_items):
    #     for j in range(n_items):
    #         pointwise_mutual_information[i][j] = (
    #                 (n_co_occurences[i][j] * n_users) /
    #                 (n_co_occurences[i].sum() * n_co_occurences[:, j].sum())
    #         )
    #         print(f'{(i / n_items) * 100 : 2.2f}% ({(j / n_items) * 100 : 2.2f}%)')

    # Calculate the SPPMI for some k
    k = 1
    pointwise_mutual_information = np.clip(pointwise_mutual_information - np.log(k), 0, np.inf)
    return pointwise_mutual_information


def load_data():
    train_ratings, test_ratings, n_users, n_items = load_movielens_ratings()
    # sppmi = co_occurrence_sppmi(train_ratings, n_users, n_items)

    return train_ratings, test_ratings, n_users, n_items


def als_step(latent_vectors, locked_vectors, ratings, regularisation, vector_type='user'):
    if vector_type == 'user':
        YTY = locked_vectors.T.dot(locked_vectors)
        lambda_i = np.eye(YTY.shape[0]) * regularisation

        for u in range(latent_vectors.shape[0]):
            latent_vectors[u, :] = solve((YTY + lambda_i), ratings[u, :].dot(locked_vectors))

    elif vector_type == 'item':
        XTX = locked_vectors.T.dot(locked_vectors)
        lambda_i = np.eye(XTX.shape[0]) * regularisation

        for i in range(latent_vectors.shape[0]):
            latent_vectors[i, :] = solve((XTX + lambda_i), ratings[:, i].T.dot(locked_vectors))

    return latent_vectors


def update_user_vectors(user_embeddings, item_embeddings, ratings, regularisation):
    coeff_matrix = item_embeddings.T.dot(item_embeddings)
    coeff_matrix_reg = np.eye(coeff_matrix.shape[0]) * regularisation

    for u in range(user_embeddings.shape[0]):
        user_embeddings[u] = solve((coeff_matrix + coeff_matrix_reg), ratings[u].dot(item_embeddings))

    return user_embeddings


def update_item_vectors(item_embeddings, user_embeddings, ratings, regularisation):
    coeff_matrix = user_embeddings.T.dot(user_embeddings)
    coeff_matrix_reg = np.eye(coeff_matrix.shape[0]) * regularisation

    for i in range(item_embeddings.shape[0]):
        item_embeddings[i] = solve(
            (coeff_matrix + coeff_matrix_reg),
            ratings[:, i].T.dot(user_embeddings)
        )

    return item_embeddings


def update_item_vectors_joint(item_embeddings, user_embeddings, context_embeddings, ratings, sppmi, regularisation):
    user_coeff_matrix = user_embeddings.T.dot(user_embeddings)
    context_coeff_matrix = context_embeddings.T.dot(context_embeddings)
    coeff_matrix_reg = np.eye(user_coeff_matrix.shape[0]) * regularisation

    for i in range(item_embeddings.shape[0]):
        item_embeddings[i] = solve(
            (user_coeff_matrix + context_coeff_matrix + coeff_matrix_reg),
            ratings[:, i].T.dot(user_embeddings) + sppmi[i].dot(context_embeddings)
        )

    return item_embeddings


def update_context_vectors(context_embeddings, item_embeddings, sppmi, regularisation):
    coeff_matrix = item_embeddings.T.dot(item_embeddings)
    coeff_matrix_reg = np.eye(coeff_matrix.shape[0]) * regularisation

    for c in range(context_embeddings.shape[0]):
        context_embeddings[c] = solve(
            (coeff_matrix + coeff_matrix_reg),
            sppmi[:, c].T.dot(item_embeddings)
        )

    return context_embeddings


def rmse(user_ratings, user_embeddings, item_embeddings):
    sse = 0
    n = 0
    for u, u_embedding in enumerate(user_embeddings):
        for i, i_embedding in enumerate(item_embeddings):
            r = user_ratings[u][i]
            if r > 0:
                sse += (r - u_embedding @ i_embedding) ** 2
                n += 1

    return np.sqrt(sse / n)


def mean_average_precision(user_ratings, user_embeddings, item_embeddings, n):
    average_precisions = []
    for u, u_embedding in enumerate(user_embeddings):
        ratings = user_ratings[u].nonzero()[0]
        item_similarities = item_embeddings @ u_embedding
        sorted_items = list(sorted(enumerate(item_similarities), key=lambda x: x[1], reverse=True))
        top_n = [m for m, s in sorted_items[:n]]

        precisions = []
        for i in range(1, n+1):
            seen = top_n[:i]
            tp = np.sum([1 for m in seen if m in ratings])
            precisions.append(tp / i)

        average_precisions.append(np.mean(precisions))

    return np.mean(average_precisions)


def MF():
    train_ratings, test_ratings, n_users, n_items = load_data()
    train_matrix = ratings_matrix(train_ratings, n_users, n_items)
    test_matrix = ratings_matrix(test_ratings, n_users, n_items)

    K = 25  # n latent factors

    U = np.random.rand(n_users, K)  # User embeddings
    I = np.random.rand(n_items, K)  # Item embeddings
    Y = np.random.rand(n_items, K)  # (Context) item embeddings

    U_reg = 0.01
    I_reg = 0.01
    Y_reg = 0.01

    n_iter = 25

    n = 20

    to_save = {
        'training': {
            'rmse': [],
            'map': [],
            'n': n
        },
        'testing': {
            'rmse': [],
            'map': [],
            'n': n
        },

        'name': "Matrix Factorization",
        'f_name': f"matrix_factorization_{n_iter}_iter_{K}_k.json",
        'trained_with': "Alternating Least Squares",
        'n_iterations': n_iter,
        'n_latent_factors': K,
        'user_reg': U_reg,
        'item_reg': I_reg
    }

    def evaluate():
        train_rmse = rmse(train_matrix, U, I)
        test_rmse = rmse(test_matrix, U, I)
        train_map = mean_average_precision(train_matrix, U, I, n)
        test_map = mean_average_precision(test_matrix, U, I, n)

        to_save['training']['rmse'].append(train_rmse)
        to_save['testing']['rmse'].append(test_rmse)
        to_save['training']['map'].append(train_map)
        to_save['testing']['map'].append(test_map)

        print(f'Evaluating at iteration {iteration}:')
        print(f'    Train RMSE: {train_rmse}')
        print(f'    Test RMSE:  {test_rmse}')
        print(f'')
        print(f'    Train MAP@{n}: {train_map}')
        print(f'    Test MAP@{n}:  {test_map}')

    for iteration in range(n_iter):
        evaluate()

        print(f'Updating user embeddings...')
        U = update_user_vectors(U, I, train_matrix, U_reg)

        print(f'Updating item embeddings...')
        I = update_item_vectors(I, U, train_matrix, I_reg)

    save_path = f'../results/matrix_factorization/{to_save["f_name"]}'
    print(f'Writing data to {save_path}...')
    with open(save_path, 'w') as fp:
        json.dump(to_save, fp, indent=True)


def CoFactor():
    print('Loading data...')
    train_ratings, test_ratings, n_users, n_items = load_data()
    train_matrix = ratings_matrix(train_ratings, n_users, n_items)
    test_matrix = ratings_matrix(test_ratings, n_users, n_items)

    print('Loading SPPMI...')
    sppmi = co_occurrence_sppmi(train_ratings, n_users, n_items)

    print('Building CoFactor...')
    K = 25  # n latent factors

    U = np.random.rand(n_users, K)  # User embeddings
    I = np.random.rand(n_items, K)  # Item embeddings
    Y = np.random.rand(n_items, K)  # (Context) item embeddings

    U_reg = 0.01
    I_reg = 0.01
    Y_reg = 0.01

    n_iter = 25

    n = 20

    to_save = {
        'training': {
            'rmse': [],
            'map': [],
            'n': n
        },
        'testing': {
            'rmse': [],
            'map': [],
            'n': n
        },

        'name': "Explicit CoFactor",
        'f_name': f"cofactor_{n_iter}_iter_{K}_k.json",
        'trained_with': "Alternating Least Squares",
        'n_iterations': n_iter,
        'n_latent_factors': K,
        'user_reg': U_reg,
        'item_reg': I_reg,
        'context_reg': Y_reg
    }

    def evaluate():
        train_rmse = rmse(train_matrix, U, I)
        test_rmse = rmse(test_matrix, U, I)
        train_map = mean_average_precision(train_matrix, U, I, n)
        test_map = mean_average_precision(test_matrix, U, I, n)

        to_save['training']['rmse'].append(train_rmse)
        to_save['testing']['rmse'].append(test_rmse)
        to_save['training']['map'].append(train_map)
        to_save['testing']['map'].append(test_map)

        print(f'Evaluating at iteration {iteration}:')
        print(f'    Train RMSE: {train_rmse}')
        print(f'    Test RMSE:  {test_rmse}')
        print(f'')
        print(f'    Train MAP@{n}: {train_map}')
        print(f'    Test MAP@{n}:  {test_map}')

    for iteration in range(n_iter):
        evaluate()

        print(f'Updating user embeddings...')
        U = update_user_vectors(U, I, train_matrix, U_reg)

        print(f'Updating item embeddings...')
        I = update_item_vectors_joint(I, U, Y, train_matrix, sppmi, I_reg)

        print(f'Updating context embeddings...')
        Y = update_context_vectors(Y, I, sppmi, Y_reg)

    save_path = f'../results/cofactor/{to_save["f_name"]}'
    print(f'Writing data to {save_path}...')
    with open(save_path, 'w') as fp:
        json.dump(to_save, fp, indent=True)


if __name__ == '__main__':
    # MF()
    CoFactor()
