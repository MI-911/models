import numpy as np
from numpy.linalg import solve, inv
import json
import pandas as pd
from random import shuffle
import random
from models.cofactor import (
    co_occurrence_sppmi,
    update_context_vectors,
    update_item_vectors,
    update_item_vectors_joint,
    update_item_vectors_joint_v2)


LIKE = 'LIKE'
DISLIKE = 'DISLIKE'
UNKNOWN = 'UNKNOWN'


def split(lst, ratio):
    split_index = int(len(lst) * ratio)
    first = lst[:split_index]
    second = lst[split_index:]
    return first, second


def user_major_ratings(ratings):
    user_ratings = {}
    for u, m, r in ratings:
        if u not in user_ratings:
            user_ratings[u] = []
        user_ratings[u].append((m, r))

    return user_ratings


def categorize(rating):
    if rating > 3:
        return LIKE
    elif rating == 0:
        return UNKNOWN
    else:
        return DISLIKE


class TrainingUser:
    def __init__(self, index):
        self.index = index
        self.ratings = []

    def instantiate(self):
        ratings = self.ratings
        self.ratings = {m: r for m, r in ratings}

    def ask(self, m):
        return categorize(self.ratings[m]) if m in self.ratings else 0


class TestingUser:
    def __init__(self, index):
        self.index = index
        self.ratings = []

        self.interview_answers = {}
        self.post_interview_answers = {}

    def instantiate(self):
        shuffle(self.ratings)
        train, test = split(self.ratings, 0.75)
        self.interview_answers = {m: r for m, r in train}
        self.post_interview_answers = {m: r for m, r in test}

    def ask(self, m, interviewing=True):
        if interviewing:
            if m in self.interview_answers:
                return categorize(self.interview_answers[m])
            else:
                return UNKNOWN
        else:
            if m in self.post_interview_answers:
                return categorize(self.post_interview_answers[m])
            else:
                return UNKNOWN


def load_movielens_ratings():
    with open('../data/movielens/ratings.csv') as fp:
        df = pd.read_csv(fp)
        ratings = [(int(u), int(m), int(r)) for u, m, r in df[['userId', 'movieId', 'rating']].values]

    # For every item, count its number of ratings
    m_ratings_counts = {}
    for u, m, r in ratings:
        if m not in m_ratings_counts:
            m_ratings_counts[m] = 0

    # We only want the top-n rated items.
    n = 100
    top_n_items = list(sorted(m_ratings_counts.items(), key=lambda x: x[1], reverse=True))[:n]
    top_n_items = [m for m, s in top_n_items]
    ratings = [(u, m, r) for u, m, r in ratings if m in top_n_items]

    # For every user, count their number of ratings.
    u_ratings_counts = {}
    for u, m, r in ratings:
        if u not in u_ratings_counts:
            u_ratings_counts[u] = 0
        u_ratings_counts[u] += 1

    print(f'Loaded {len(u_ratings_counts)} users. Filtering for at least 5 ratings...')

    # We only want users with >5 ratings.
    ratings = [(u, m, r) for u, m, r in ratings if u_ratings_counts[u] > 5]

    # Map users and items to indices
    u_idx_map, uc = {}, 0
    m_idx_map, mc = {}, 0
    for u, m, r in ratings:
        if u not in u_idx_map:
            u_idx_map[u] = uc
            uc += 1
        if m not in m_idx_map:
            m_idx_map[m] = mc
            mc += 1

    print(f'After filtering, {mc} items and {uc} users in dataset.')

    # Indexize the ratings
    ratings = [(u_idx_map[u], m_idx_map[m], r) for u, m, r in ratings]

    # Store ratings in user_major
    user_ratings = user_major_ratings(ratings)

    # Shuffle the user indices and split
    user_indices = list(user_ratings.keys())
    shuffle(user_indices)
    train_user_indices, test_user_indices = split(user_indices, 0.75)

    # Create user objects from them
    train_users = [TrainingUser(idx) for idx in train_user_indices]
    test_users = [TestingUser(idx) for idx in test_user_indices]

    # Put their ratings in the objects
    for u in train_users:
        u.ratings = user_ratings[u.index]
    for u in test_users:
        u.ratings = user_ratings[u.index]

    # Instantiate their answer sets
    [u.instantiate() for u in train_users]
    [u.instantiate() for u in test_users]

    # Return
    return train_users, test_users, uc, mc, len(ratings)


def ratings_matrix(users, n_users, n_items):
    M = np.zeros((n_users, n_items))
    for u in users:
        for m, r in u.ratings.items():
            M[u.index][m] = r

    return M


def ratings_triples(users):
    triples = []
    for u in users:
        for m, r in u.ratings.items():
            triples.append((u, m, r))

    return triples


def liked_triples(users):
    triples = []
    for u in users:
        for m, r in u.ratings.items():
            if categorize(r) == LIKE:
                triples.append((u, m, r))

    return triples


def disliked_triples(users):
    triples = []
    for u in users:
        for m, r in u.ratings.items():
            if categorize(r) == DISLIKE:
                triples.append((u, m, r))

    return triples


def items_in_user_group(users):
    items = []
    for u in users:
        items += [m for m, r in u.ratings.items()]

    return set(items)


def split_users(users, q):
    likes, dislikes, unknown = [], [], []
    for user in users:
        answer = user.ask(q)
        add_to = likes if answer == LIKE else dislikes if answer == DISLIKE else unknown
        add_to.append(user)

    return likes, dislikes, unknown


def optimal_profile(users, default_profile=None):
    if len(users) == 0:
        return default_profile

    coeff_matrix = np.zeros((K, K))
    coeff_matrix_reg = np.eye(K) * U_reg
    coeff_vector = np.zeros(K)

    # For calculating the loss, stack item embeddings and ratings
    ratings = []
    embeddings = []

    for u in users:
        for m, r in u.ratings.items():
            i_embedding = I[m]
            i_embedding_v = i_embedding.reshape((1, K))
            coeff_matrix += i_embedding_v.T.dot(i_embedding_v) + coeff_matrix_reg
            coeff_vector += i_embedding * r

            ratings.append(r)
            embeddings.append(i_embedding)

    try:
        profile = inv(coeff_matrix).dot(coeff_vector)
    except np.linalg.LinAlgError as e:
        print(e)
        profile = default_profile

    # Calculate the loss
    ratings = np.array(ratings)
    embeddings = np.array(embeddings)
    profiles = np.ones_like(embeddings) * profile

    predictions = np.sum(profiles * embeddings, axis=1)
    return profile, np.sum((predictions - ratings) ** 2)


class Node:
    def __init__(self, users, profile, depth, max_depth):
        self.users = users
        self.profile = profile
        self.depth = depth
        self.max_depth = max_depth
        self.q = np.nan
        self.like, self.dislike, self.unknown = None, None, None

    def split(self):

        if self.depth >= self.max_depth:
            return self

        chosen_q = np.nan
        lowest_loss = np.inf

        best_groups = None
        best_profiles = None

        for q in question_set:
            likes, dislikes, unknown = split_users(self.users, q)
            p_likes, likes_loss = optimal_profile(likes, default_profile=self.profile)
            p_dislikes, dislikes_loss = optimal_profile(dislikes, default_profile=self.profile)
            p_unknown, unknown_loss = optimal_profile(unknown, default_profile=self.profile)

            loss = likes_loss + dislikes_loss + unknown_loss
            if loss < lowest_loss:
                lowest_loss = loss
                chosen_q = q
                best_groups = likes, dislikes, unknown
                best_profiles = p_likes, p_dislikes, p_unknown

        likes, dislikes, unknown = best_groups
        p_likes, p_dislikes, p_unknown = best_profiles

        self.like = Node(likes, p_likes, self.depth + 1, self.max_depth)
        self.dislike = Node(dislikes, p_dislikes, self.depth + 1, self.max_depth)
        self.unknown = Node(unknown, p_unknown, self.depth + 1, self.max_depth)

        self.q = chosen_q
        question_set.remove(chosen_q)

        return self

    def interview(self, user, interviewing=True):
        if self.q == np.nan:
            return self.profile  # We didn't split

        if isinstance(user, TrainingUser):
            answer = user.ask(self.q)
            if answer == LIKE:
                if self.like is None:
                    return self.profile
                return self.like.interview(user)
            elif answer == DISLIKE:
                if self.dislike is None:
                    return self.profile
                return self.dislike.interview(user)
            else:
                if self.unknown is None:
                    return self.profile
                return self.unknown.interview(user)

        elif isinstance(user, TestingUser):
            answer = user.ask(self.q, interviewing)
            if answer == LIKE:
                if self.like is None:
                    return self.profile
                return self.like.interview(user, interviewing)
            elif answer == DISLIKE:
                if self.dislike is None:
                    return self.profile
                return self.dislike.interview(user, interviewing)
            else:
                if self.unknown is None:
                    return self.profile
                return self.unknown.interview(user, interviewing)


def interview_users(users, tree):
    for u in users:
        U[u.index] = tree.interview(u)

    return U


def update_item_vectors_low_rmse(users):
    for m in range(n_items):
        coeff_matrix = np.zeros((K, K))
        coeff_matrix_reg = np.eye(K) * 0.0015
        coeff_vector = np.zeros(K)

        # Find the users who have seen this item
        users = [u for u in users if m in u.ratings]

        for u in users:
            r = u.ratings[m]
            Ta = U[u.index].reshape((1, K))
            coeff_matrix += Ta.T @ Ta + coeff_matrix_reg
            coeff_vector += r * Ta.reshape(K)

        coeff_matrix = inv(coeff_matrix)
        I[m] = coeff_matrix.dot(coeff_vector)


def update_item_vectors_joint_low_rmse(users):
    for m in range(n_items):
        user_coeff_matrix = np.zeros((K, K))
        coeff_matrix_reg = np.eye(K) * 0.0015
        user_coeff_vector = np.zeros(K)

        # Find the users who have seen this item
        users = [u for u in users if m in u.ratings]

        for u in users:
            r = u.ratings[m]
            Ta = U[u.index].reshape((1, K))
            user_coeff_matrix += Ta.T @ Ta + coeff_matrix_reg
            user_coeff_vector += r * Ta.reshape(K)

        context_coeff_matrix = Y.T.dot(Y)
        context_coeff_vector = sppmi[m].dot(Y)

        coeff_matrix = inv(user_coeff_matrix + context_coeff_matrix)
        I[m] = coeff_matrix.dot(user_coeff_vector + context_coeff_vector)


def rmse(users, training=True):
    sse = 0
    n = 0

    for u in users:
        u_embedding = U[u.index]
        if training:
            for m, r in u.ratings.items():
                i_embedding = I[m]
                sse += (r - u_embedding.dot(i_embedding)) ** 2
                n += 1
        else:
            for m, r in u.post_interview_answers.items():
                i_embedding = I[m]
                sse += (r - u_embedding.dot(i_embedding)) ** 2
                n += 1

    return np.sqrt(sse / n)


def mean_average_precision(users, n, training=True):
    average_precisions = []

    for u in users:
        u_embedding = U[u.index]
        item_similarities = I @ u_embedding
        sorted_items = list(sorted(enumerate(item_similarities), key=lambda x: x[1], reverse=True))
        top_n = [m for m, s in sorted_items[:n]]

        precisions = []
        for i in range(1, n + 1):
            seen = top_n[:i]
            tp = np.sum([1 for m in seen if u.ask(m) == LIKE]) if training else \
                np.sum([1 for m in seen if u.ask(m, interviewing=False) == LIKE])
            precisions.append(tp / i)

        average_precisions.append(np.mean(precisions))

    return np.mean(average_precisions)


def FMF():
    # Training settings
    n_iter = 10
    n = 20  # MAP@n
    max_depth = 5

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

        'name': "Functional Matrix Factorization",
        'trained_with': 'Alternating Least Squares',
        'notes': 'Updated item vectors w.r.t. all user vectors, regardless of whether they had rated the item or not. This naturally causes a higher RMSE, but much better MAP, too',
        'f_name': f"fmf_broad_updates_{n_iter}_iter_{K}_k.json",
        'n_iterations': n_iter,
        'n_latent_features': K,
        'max_depth': max_depth,
        'user_reg': U_reg,
        'item_reg': I_reg
    }

    def evaluate():
        train_rmse = rmse(train_users, training=True)
        test_rmse = rmse(test_users, training=False)
        train_map = mean_average_precision(train_users, n, training=True)
        test_map = mean_average_precision(test_users, n, training=False)

        to_save['training']['rmse'].append(train_rmse)
        to_save['testing']['rmse'].append(test_rmse)
        to_save['training']['map'].append(train_map)
        to_save['testing']['map'].append(test_map)

        print(f'Evaluating at iteration {iteration}:')
        print(f'    Train RMSE: {train_rmse}')
        print(f'    Test RMSE: {test_rmse}')
        print('')
        print(f'    Train MAP@{n}: {train_map}')
        print(f'    Test MAP@{n}:  {test_map}')

    root = Node(train_users, optimal_profile(train_users), depth=0, max_depth=max_depth)

    for iteration in range(n_iter):

        evaluate()

        # Build the tree
        print('Building tree...')
        tree = root.split()

        # Update user embeddings
        print('Updating user embeddings')
        interview_users(train_users, tree)
        interview_users(test_users, tree)

        # Update item embeddings
        print('Updating item embeddings')
        # update_item_vectors_low_rmse(train_users)  # Low RMSE, shitty MAP
        update_item_vectors(I, U, train_matrix, I_reg)

    file_path = f'../results/functional_matrix_factorization/{to_save["f_name"]}'
    print(f'Writing to disk at {file_path}...')
    with open(file_path, 'w') as fp:
        json.dump(to_save, fp)


def CoFactorFMF():
    # Training settings
    n_iter = 10
    n = 20  # MAP@n
    max_depth = 5

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

        'name': "Functional Matrix Factorization ft. CoFactor",
        'trained_with': 'Alternating Least Squares (Joint Learning on item embeddings)',
        'notes': 'Updating item embeddings broadly, using co-liked and co-disliked SPPMI.',
        'f_name': f"fmf_cofactor_broad_updates_co_liked_disliked_{n_iter}_iter_{K}_k.json",
        'n_iterations': n_iter,
        'n_latent_features': K,
        'max_depth': max_depth,
        'user_reg': U_reg,
        'item_reg': I_reg
    }

    def evaluate():
        train_rmse = rmse(train_users, training=True)
        test_rmse = rmse(test_users, training=False)
        train_map = mean_average_precision(train_users, n, training=True)
        test_map = mean_average_precision(test_users, n, training=False)

        to_save['training']['rmse'].append(train_rmse)
        to_save['testing']['rmse'].append(test_rmse)
        to_save['training']['map'].append(train_map)
        to_save['testing']['map'].append(test_map)

        print(f'Evaluating at iteration {iteration}:')
        print(f'    Train RMSE: {train_rmse}')
        print(f'    Test RMSE: {test_rmse}')
        print('')
        print(f'    Train MAP@{n}: {train_map}')
        print(f'    Test MAP@{n}:  {test_map}')

    root = Node(train_users, optimal_profile(train_users), depth=0, max_depth=max_depth)

    for iteration in range(n_iter):
        evaluate()

        # Build the tree
        print('Building tree...')
        tree = root.split()

        # Update user embeddings
        print('Updating user embeddings')
        interview_users(train_users, tree)
        interview_users(test_users, tree)

        # Update item embeddings
        print('Updating item embeddings')
        # update_item_vectors_low_rmse(train_users)  # Low RMSE, shitty MAP  # TODO: This is narrow-update FMF
        # update_item_vectors_joint_low_rmse(train_users)  # TODO: This is narrow_update FMF w. CoFactor
        update_item_vectors_joint_v2(I, U, Y, train_matrix, sppmi_liked, sppmi_disliked, I_reg)  # TODO: This is broad-update FMF
        update_context_vectors(Y, I, sppmi, Y_reg)  # TODO: This is broad-update FMF w. CoFactor

    file_path = f'../results/functional_matrix_factorization/{to_save["f_name"]}'
    print(f'Writing to disk at {file_path}...')
    with open(file_path, 'w') as fp:
        json.dump(to_save, fp)


if __name__ == "__main__":
    train_users, test_users, n_users, n_items, n_ratings = load_movielens_ratings()
    train_matrix = ratings_matrix(train_users, n_users, n_items)
    train_ratings = ratings_triples(train_users)
    train_liked_ratings = liked_triples(train_users)
    train_disliked_ratings = disliked_triples(train_users)
    sppmi = co_occurrence_sppmi(train_ratings, n_users, n_items)
    sppmi_liked = co_occurrence_sppmi(train_liked_ratings, n_users, n_items)
    sppmi_disliked = co_occurrence_sppmi(train_disliked_ratings, n_users, n_items)

    # Hyper params
    K = 25

    U = np.random.rand(n_users, K)
    I = np.random.rand(n_items, K)
    Y = np.random.rand(n_items, K)

    U_reg = 0.01
    I_reg = 0.01
    Y_reg = 0.01

    # What questions can be asked about
    question_set = set([i for i in range(n_items)])

    CoFactorFMF()









