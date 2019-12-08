from os.path import join
from data.cold_start import get_label_map, get_top_entities
import json
import numpy as np
import random


class User:
    def __init__(self, idx, n_movies, n_entities):
        self.n_movies = n_movies
        self.n_entities = n_entities
        self.idx = idx

        self.movie_ratings = []
        self.entity_ratings = []

        self.lv = np.zeros(n_movies + n_entities)
        self.dv = np.zeros(n_movies + n_entities)

        self.interview_answers = {}
        self.post_interview_answers = {}

    def warm_start(self, include_entity_ratings=False):
        for o, r in self.movie_ratings + self.entity_ratings if include_entity_ratings else self.movie_ratings:
            if r == 1:
                self.lv[o] = r
            elif r == -1:
                self.dv[o] = r

    def cold_start(self):
        random.shuffle(self.movie_ratings)

        train_movie_ratings, test_movie_ratings = split(self.movie_ratings)
        self.interview_answers = {o: r for o, r in train_movie_ratings + self.entity_ratings}
        self.post_interview_answers = {m: r for m, r in test_movie_ratings}

    def ask(self, item):
        answer = self.lv[item]
        if answer == 0:
            answer = self.dv[item]
        return answer

    def interview(self, item):
        if item in self.interview_answers:
            return self.interview_answers[item]
        else:
            return 0

    def post_interview(self, item):
        if item in self.post_interview_answers:
            return self.post_interview_answers[item]
        else:
            return 0


def generate_dataset(mindreader_dir='../data/mindreader', top_n=100):
    """
    Lorem ipsum
    :param mindreader_dir:
    :param top_n:
    :return:
    """
    with open(join(mindreader_dir, 'ratings_clean.json')) as fp:
        all_ratings = json.load(fp)
    with open(join(mindreader_dir, 'entities_clean.json')) as fp:
        entities = json.load(fp)

    # Filter out "don't know" ratings
    # TODO: See if we can use them for something
    all_ratings = [(uid, uri, r) for uid, uri, r in all_ratings if not r == 0]

    label_map = get_label_map(entities)

    # Split into movie and entity ratings
    movie_ratings = [(uid, uri, r) for uid, uri, r in all_ratings if 'Movie' in label_map[uri]]
    entity_ratings = [(uid, uri, r) for uid, uri, r in all_ratings if 'Movie' not in label_map[uri]]

    # Take the top-N
    top_movies = get_top_entities(movie_ratings, top_n=top_n) if top_n is not None else movie_ratings
    top_entities = get_top_entities(entity_ratings, top_n=top_n) if top_n is not None else entity_ratings

    # Filter out non-top entity/movie ratings
    movie_ratings = [(uid, uri, r) for uid, uri, r in all_ratings if uri in top_movies]
    entity_ratings = [(uid, uri, r) for uid, uri, r in all_ratings if uri in top_entities]

    # Map UIDs and URIs to indices and back again
    uid_idx_map = {}
    idx_uid_map = {}

    m_uri_idx_map = {}
    m_idx_uri_map = {}

    e_uri_idx_map = {}
    e_idx_uri_map = {}
    u, m, e = 0, 0, 0

    for uid, uri, r in movie_ratings:
        if uid not in uid_idx_map:
            uid_idx_map[uid] = u
            idx_uid_map[u] = uid
            u += 1
        if uri not in m_uri_idx_map:
            m_uri_idx_map[uri] = m
            m_idx_uri_map[m] = uri
            m += 1
    for uid, uri, r in entity_ratings:
        if uid not in uid_idx_map:
            uid_idx_map[uid] = u
            idx_uid_map[u] = uid
            u += 1
        if uri not in e_uri_idx_map:
            e_uri_idx_map[uri] = e
            e_idx_uri_map[e] = uri
            e += 1

    # Entity URI/index maps are not corrected for movie indices, so correct that
    _e_uri_idx_map = {}
    _e_idx_uri_map = {}
    for uri, idx in e_uri_idx_map.items():
        _e_uri_idx_map[uri] = idx + m
        _e_idx_uri_map[idx + m] = uri
    e_uri_idx_map = _e_uri_idx_map
    e_idx_uri_map = _e_idx_uri_map

    # Create indexed rating triples
    movie_ratings = [
        (uid_idx_map[uid], m_uri_idx_map[uri], r)
        for uid, uri, r
        in movie_ratings
    ]

    entity_ratings = [
        (uid_idx_map[uid], e_uri_idx_map[uri], r)
        for uid, uri, r
        in entity_ratings
    ]

    # Create the users
    user_set = [User(idx, m, e) for idx in range(u)]

    # Assign ratings to users
    for u_idx, m_idx, r in movie_ratings:
        user = user_set[u_idx]
        user.movie_ratings.append((m_idx, r))

    for u_idx, e_idx, r in entity_ratings:
        user = user_set[u_idx]
        user.movie_ratings.append((e_idx, r))

    random.shuffle(user_set)

    return user_set, u, m, e


def split(lst, ratio=0.75):
    split_index = int(len(lst) * ratio)
    first = lst[:split_index]
    second = lst[:split_index]
    return first, second


def split_users(users, item):
    L, D, U = [], [], []
    for u in users:
        a = u.ask(item)
        if a == 1:
            L.append(u)
        elif a == -1:
            D.append(u)
        else:
            U.append(u)

    return L, D, U


def mean_intersection_vector(users):
    if len(users) == 0:
        return np.zeros(n_movies + n_entities)
    stacked_like_vectors = np.stack([u.lv for u in users])
    return np.sum(stacked_like_vectors, axis=0) / len(users)


class Node:
    def __init__(self, users, mean_intersection_vector, depth):
        self.users = users
        self.depth = depth
        self.mean_intersection_vector = mean_intersection_vector
        self.split_item = None

        self.L_child, self.D_child, self.U_child = None, None, None

    def split(self, max_depth):
        if self.depth >= max_depth:
            print(f'Stopping at max depth')
            return

        best_split_item = None
        best_mean_intersection = 0
        for o in SPLIT_ENTITIES:
            L, D, U = split_users(self.users, o)
            sum_mean_intersections = np.sum([np.sum(mean_intersection_vector(_us)) for _us in [L, D, U]])
            if sum_mean_intersections > best_mean_intersection:
                best_split_item = o
                best_mean_intersection = sum_mean_intersections

        SPLIT_ENTITIES.remove(best_split_item)
        self.split_item = best_split_item
        L, D, U = split_users(self.users, best_split_item)
        self.L_child = Node(L, mean_intersection_vector(L), self.depth + 1) if len(L) > 5 else None
        self.D_child = Node(D, mean_intersection_vector(D), self.depth + 1) if len(D) > 5 else None
        self.U_child = Node(U, mean_intersection_vector(U), self.depth + 1) if len(U) > 5 else None

        # Split child nodes
        if self.L_child:
            print(f'Splitting node with {len(L)} users at depth {self.depth}')
            self.L_child.split(max_depth)
        if self.D_child:
            print(f'Splitting node with {len(D)} users at depth {self.depth}')
            self.D_child.split(max_depth)
        if self.U_child:
            print(f'Splitting node with {len(U)} users at depth {self.depth}')
            self.U_child.split(max_depth)

    def interview(self, user):
        answer = user.interview(self.split_item)
        if answer == 1:
            if self.L_child:
                return self.L_child.interview(user)
            else:
                return self.mean_intersection_vector
        elif answer == -1:
            if self.D_child:
                return self.D_child.interview(user)
            else:
                return self.mean_intersection_vector
        else:
            if self.U_child:
                return self.U_child.interview(user)
            else:
                return self.mean_intersection_vector

    def print(self):
        print(self.mean_intersection_vector)
        if self.L_child:
            self.L_child.print()
        if self.D_child:
            self.D_child.print()
        if self.U_child:
            self.U_child.print()


def average_precision(user, predictions):
    n = len(predictions)
    n_correct = 0
    pre_avg = 0

    for _i, i in enumerate(predictions):
        answer = user.post_interview(i)
        if answer == 1:
            n_correct += 1
            pre_avg += n_correct / _i

    return pre_avg / n


if __name__ == '__main__':
    random.seed(42)
    users, n_users, n_movies, n_entities = generate_dataset(mindreader_dir='../data/mindreader/')

    # SPLIT_ENTITIES = set([i for i in range(n_movies)])  # Build tree from movies only
    # SPLIT_ENTITIES = [i for i in range(n_entities)]   # Build tree from entities only
    SPLIT_ENTITIES = [i for i in range(n_movies + n_entities)]  # Build tree from movies and entities

    train_users, test_users = split(users)

    # Initialize the user vectors
    [u.warm_start() for u in train_users]
    [u.warm_start() for u in test_users]

    root = Node(train_users, mean_intersection_vector(train_users), depth=0)
    root.split(max_depth=3)  # Build the tree

    # Evaluate on test users
    aps = []
    for u in test_users:
        predictions = root.interview(u)
        sorted_predictions = list(sorted([(i, r) for i, r in enumerate(predictions)], key=lambda x: x[1], reverse=True))
        like_predictions = [i for i, r in sorted_predictions[:100]]  # AP@100
        aps.append(average_precision(u, like_predictions))

    print(f'MAP@20: {np.mean(aps)}')











