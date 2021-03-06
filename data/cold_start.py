import numpy as np
import pandas as pd
import json
from os.path import join
from random import shuffle


LIKE = 1
DISLIKE = -1
UNKNOWN = 0


class User:
    def __init__(self, idx, movie_ratings=[], entity_ratings=[], split_ratio=0.75):
        """
        A simulated user. Provided movie and entity ratings, can dynamically shuffle and
        generate training/test sets of answers.
        :param idx: The index of the user.
        :param movie_ratings: (Optional) A list of (movie_idx, rating) pairs.
        :param entity_ratings: (Optional) A list of (entity_idx, rating) pairs.
        :param split_ratio: (Optional, default 0.75) the ratio between training/test set sizes.
        """

        self.idx = idx
        self.movie_ratings = movie_ratings
        self.entity_ratings = entity_ratings
        self.split_ratio = split_ratio

        self.m_answers = None
        self.e_answers = None
        self.m_test_answers = None
        self.e_test_answers = None

    def shuffle(self):
        """
        Shuffles the user's ratings and generates new training and test answer sets.
        """
        self._reset()

        # Shuffle the ratings
        shuffle(self.movie_ratings)
        shuffle(self.entity_ratings)

        # Split into training and test answers
        m_answers, m_test_answers = split(self.movie_ratings, split_ratio=self.split_ratio)
        e_answers, e_test_answers = split(self.entity_ratings, split_ratio=self.split_ratio)

        self.m_answers = {m: r for m, r in m_answers}
        self.e_answers = {e: r for e, r in e_answers}
        self.m_test_answers = {m: r for m, r in m_test_answers}
        self.e_test_answers = {e: r for e, r in e_test_answers}

    def ask_movie(self, m, evaluation=False):
        answers = self.m_test_answers if evaluation else self.m_answers
        return answers[m] if m in answers else UNKNOWN

    def ask_entity(self, e, evaluation=False):
        answers = self.e_test_answers if evaluation else self.e_answers
        return answers[e] if e in answers else UNKNOWN

    def _reset(self):
        self.m_answers = {}
        self.e_answers = {}
        self.m_test_answers = {}
        self.e_test_answers = {}


class DataSet:
    def __init__(self, user_set, n_users, n_movies, n_entities):
        self.user_set = user_set
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_entities = n_entities

    def split_users(self, split_ratio=0.75):
        return split(self.user_set, split_ratio=split_ratio)

    @staticmethod
    def shuffle(user_set):
        shuffle(user_set)  # Shuffle the order of users
        [u.shuffle() for u in user_set]  # Shuffle ratings for each user, generate new learning and test sets
        return user_set


def get_label_map(entities):
    # Map URIs to labels
    label_map = {}
    for uri, name, labels in entities:
        labels = labels.split('|')
        if uri not in label_map:
            label_map[uri] = labels

    return label_map


def get_top_entities(ratings, top_n):
    count_map = {}
    for uid, uri, r in ratings:
        if uri not in count_map:
            count_map[uri] = 0
        count_map[uri] += 1

    sorted_entities = sorted(count_map.items(), key=lambda x: x[1], reverse=True)
    sorted_entities = [uri for uri, count in sorted_entities]
    return sorted_entities if top_n is None else sorted_entities[:top_n]


def split(lst, split_ratio=0.75):
    split_index = int(len(lst) * split_ratio)
    lst1 = lst[:split_index]
    lst2 = lst[split_index:]

    return lst1, lst2


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
    top_movies = get_top_entities(movie_ratings, top_n=top_n)
    top_entities = get_top_entities(entity_ratings, top_n=top_n)

    # Filter out non-top entity/movie ratings
    movie_ratings = [(uid, uri, r) for uid, uri, r in all_ratings if uri in top_movies]
    entity_ratings = [(uid, uri, r) for uid, uri, r in all_ratings if uri in top_entities]

    # Map UIDs and URIs to indices
    uid_idx_map = {}
    m_uri_idx_map = {}
    e_uri_idx_map = {}
    u, m, e = 0, 0, 0

    for uid, uri, r in movie_ratings:
        if uid not in uid_idx_map:
            uid_idx_map[uid] = u
            u += 1
        if uri not in m_uri_idx_map:
            m_uri_idx_map[uri] = m
            m += 1
    for uid, uri, r in entity_ratings:
        if uid not in uid_idx_map:
            uid_idx_map[uid] = u
            u += 1
        if uri not in e_uri_idx_map:
            e_uri_idx_map[uri] = e
            e += 1

    # Create indexed rating triples
    movie_ratings = [
        (uid_idx_map[uid], m_uri_idx_map[uri], r)
        for uid, uri, r
        in movie_ratings
    ]

    entity_ratings = [
        (uid_idx_map[uid], e_uri_idx_map[uri] + m, r)  # Add movie idx to convert to same index space
        for uid, uri, r
        in entity_ratings
    ]

    # Create the users
    user_set = [User(idx) for idx in range(u)]

    # Assign ratings to users
    for u_idx, m_idx, r in movie_ratings:
        user_set[u_idx].movie_ratings.append((m_idx, r))
    for u_idx, e_idx, r in entity_ratings:
        user_set[u_idx].entity_ratings.append((e_idx, r))

    # Return the data set
    return DataSet(user_set, u, m, e)



