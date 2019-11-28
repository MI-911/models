import numpy as np
from data.training import warm_start
from random import shuffle
import json

LIKE = 1
DISLIKE = 0
UNKNOWN = -1


class User:
    def __init__(self, id):
        self.id = id
        self.answer_set = {}
        self.evaluation_set = {}
        self.movie_answers = []
        self.entity_answers = []

    def ask(self, entity):
        # Can be both a movie or an entity
        if entity in self.answer_set:
            return self.answer_set[entity]
        return UNKNOWN

    def add_movie(self, m, r):
        self.movie_answers.append((m, r))

    def add_entity(self, e, r):
        self.entity_answers.append((e, r))

    def create_answer_eval_sets(self):
        # 75% of movie ratings to answer, 25% to eval
        answers, evaluation = split(self.movie_answers, 0.75)
        answers += self.entity_answers

        self.answer_set = {e: r for e, r in answers}
        self.evaluation_set = {m: r for m, r in evaluation}


def get_label_map(entities):
    # Map URIs to labels
    label_map = {}
    for uri, name, labels in entities:
        labels = labels.split('|')
        if uri not in label_map:
            label_map[uri] = labels

    return label_map


def top_n(count_dict, n):
    sorted_by_count = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
    return [e for e, r in sorted_by_count][:n]


def split(l, ratio):
    l = list(l)
    shuffle(l)
    split_idx = int(len(l) * ratio)
    train = l[:split_idx]
    test = l[split_idx:]
    return train, test


def load_fmf_users():
    with open('../data/mindreader/ratings_clean.json') as fp:
        all_ratings = json.load(fp)
    with open('../data/mindreader/entities_clean.json') as fp:
        entities = json.load(fp)

    # Filter out "don't know" ratings
    all_ratings = [(uid, uri, r) for uid, uri, r in all_ratings if not r == 0]

    label_map = get_label_map(entities)

    # Split into movie and entity ratings
    movie_ratings = [(uid, uri, r) for uid, uri, r in all_ratings if 'Movie' in label_map[uri]]
    entity_ratings = [(uid, uri, r) for uid, uri, r in all_ratings if 'Movie' not in label_map[uri]]

    # Count the number of ratings for every entity and movie.
    # Filter out all entities and movies that are not in the top 100.
    m_c_map = {}
    e_c_map = {}
    for u, m, r in movie_ratings:
        if m not in m_c_map:
            m_c_map[m] = 0
        m_c_map[m] += 1
    for u, e, r in entity_ratings:
        if e not in e_c_map:
            e_c_map[e] = 0
        e_c_map[e] += 1

    top_movies = top_n(m_c_map, n=100)
    top_entities = top_n(e_c_map, n=100)

    # Filter out the ratings that we don't need
    movie_ratings = [(uid, uri, rating) for uid, uri, rating in movie_ratings if uri in top_movies]
    entity_ratings = [(uid, uri, rating) for uid, uri, rating in entity_ratings if uri in top_entities]

    # (Optional) count the number of movie ratings per user, filter them out as needed
    u_c_map = {}
    for u, m, r in movie_ratings:
        if u not in u_c_map:
            u_c_map[u] = 0
        u_c_map[u] += 1
    u_ids = [uid for uid, n_ratings in u_c_map.items() if n_ratings > 5]

    # Once again, filter the movie and entity ratings in case we lost some users there
    movie_ratings = [(uid, uri, rating) for uid, uri, rating in movie_ratings if uid in u_ids]
    entity_ratings = [(uid, uri, rating) for uid, uri, rating in entity_ratings if uid in u_ids]

    # Now we only have the top-100 entities and movies, and we have the users we need.
    # Start mapping the users, movies and entities to indices
    uid_idx_map, uc = {}, 0
    m_uri_idx_map, mc = {}, 0
    e_uri_idx_map, ec = {}, 0

    for u, m, r in movie_ratings:
        if u not in uid_idx_map:
            uid_idx_map[u] = uc
            uc += 1
        if m not in m_uri_idx_map:
            m_uri_idx_map[m] = mc
            mc += 1
    for u, e, r in entity_ratings:
        if u not in uid_idx_map:
            uid_idx_map[u] = uc
            uc += 1
        if e not in e_uri_idx_map:
            e_uri_idx_map[e] = ec
            ec += 1

    # Convert the triples to index notation, convert ratings
    r_convert_map = {1: 1, -1: 0}
    movie_ratings = [(uid_idx_map[u], m_uri_idx_map[m], r_convert_map[r]) for u, m, r in movie_ratings]
    entity_ratings = [(uid_idx_map[u], e_uri_idx_map[e], r_convert_map[r]) for u, e, r in entity_ratings]

    # Sample 75% of user indices and use these for training.
    # The remaining 25% goes to test.
    train_user_indices, test_user_indices = split([i for i in range(uc)], 0.75)
    train_users = {i: User(i) for i in train_user_indices}
    test_users = {i: User(i) for i in test_user_indices}

    # Give the users their answers
    # TODO: Choose to add entity ratings after testing with just movies
    for u, m, r in movie_ratings:
        if u in train_users:
            train_users[u].add_movie(m, r)
        elif u in test_users:
            test_users[u].add_movie(m, r)

    # Let the test users split their answer/evaluation sets
    [u.create_answer_eval_sets() for _, u in test_users.items()]

    # Done! Return the train and test users
    train_users = [u for _, u in train_users.items()]
    test_users = [u for _, u in test_users.items()]

    return train_users, test_users, uc, mc, ec




