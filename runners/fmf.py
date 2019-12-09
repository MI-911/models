import numpy as np
from data.training import warm_start
from random import shuffle
import json
from models.fmf import FunctionalMatrixFactorizaton
import pandas as pd

LIKE = 1
DISLIKE = -1
UNKNOWN = 0


class User:
    def __init__(self, id):
        self.id = id
        self.movie_answers = []
        self.entity_answers = []

        self.movie_evaluation = {}
        self.entity_evaluation = {}

    def ask(self, o, entity=False):
        # Can be both a movie or an entity
        if entity:
            if o in self.entity_answers:
                return self.entity_answers[o]
            else:
                return UNKNOWN

        if o in self.movie_answers:
            return self.movie_answers[o]
        else:
            return UNKNOWN

    def eval(self, o):
        # Can only be movies
        if o in self.movie_evaluation:
            return self.movie_evaluation[o]
        else:
            return UNKNOWN

    def add_movie(self, m, r):
        self.movie_answers.append((m, r))

    def add_entity(self, e, r):
        self.entity_answers.append((e, r))

    def create_answer_eval_sets(self, train=False):
        if train:
            entity_answers = self.entity_answers
            movie_answers = self.movie_answers
            self.entity_answers = {e: r for e, r in entity_answers}
            self.movie_answers = {m: r for m, r in movie_answers}
            return

        # Split the movie ratings into train and test set
        movie_answers, movie_evaluations = split(self.movie_answers, 0.75)
        self.movie_answers = {m: r for m, r in movie_answers}
        self.movie_evaluation = {m: r for m, r in movie_evaluations}
        entity_answers = self.entity_answers
        self.entity_answers = {e: r for e, r in entity_answers}

    def __repr__(self):
        return f'User {self.id} ({len(self.movie_answers) + len(self.entity_answers)} answers, {len(self.movie_evaluation)} eval answers.)'


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


def load_fmf_users(with_entities=False):
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

    m_idx_uri_map = {}
    e_idx_uri_map = {}

    for u, m, r in movie_ratings:
        if u not in uid_idx_map:
            uid_idx_map[u] = uc
            uc += 1
        if m not in m_uri_idx_map:
            m_uri_idx_map[m] = mc
            if mc not in m_idx_uri_map:
                m_idx_uri_map[mc] = m
            mc += 1
    for u, e, r in entity_ratings:
        if u not in uid_idx_map:
            uid_idx_map[u] = uc
            uc += 1
        if e not in e_uri_idx_map:
            e_uri_idx_map[e] = ec
            if ec not in e_idx_uri_map:
                e_idx_uri_map[ec] = e
            ec += 1

    # Convert the triples to index notation, convert ratings
    r_convert_map = {1: LIKE, -1: DISLIKE}
    movie_ratings = [(uid_idx_map[u], m_uri_idx_map[m], r_convert_map[r]) for u, m, r in movie_ratings]
    entity_ratings = [(uid_idx_map[u], e_uri_idx_map[e], r_convert_map[r]) for u, e, r in entity_ratings]

    # Sample 75% of user indices and use these for training.
    # The remaining 25% goes to test.
    train_user_indices, test_user_indices = split([i for i in range(uc)], 0.75)
    train_users = {i: User(i) for i in train_user_indices}
    test_users = {i: User(i) for i in test_user_indices}

    # Give the users their answers
    for u, m, r in movie_ratings:
        if u in train_users:
            train_users[u].add_movie(m, r)
        elif u in test_users:
            test_users[u].add_movie(m, r)

    if with_entities:
        for u, e, r in entity_ratings:
            if u in train_users:
                train_users[u].add_entity(e, r)
            elif u in test_users:
                test_users[u].add_entity(e, r)

    # Let the test users split their answer/evaluation sets
    [u.create_answer_eval_sets(train=True) for _, u in train_users.items()]
    [u.create_answer_eval_sets(train=False) for _, u in test_users.items()]

    # Done! Return the train and test users
    train_users = [u for _, u in train_users.items()]
    test_users = [u for _, u in test_users.items()]

    return train_users, test_users, uc, mc, ec, m_idx_uri_map, e_idx_uri_map


def load_fmf_users_movielens():
    with open('../data/movielens/ratings.csv') as fp:
        all_ratings = pd.read_csv(fp)

    all_ratings = [(uid, uri, rating) for uid, uri, rating in all_ratings[['userId', 'movieId', 'rating']].values]

    # Filter out "don't know" ratings
    movie_ratings = [(uid, uri, int(r)) for uid, uri, r in all_ratings if not r == 0]

    # Count the number of ratings for every entity and movie.
    # Filter out all entities and movies that are not in the top 100.
    m_c_map = {}
    e_c_map = {}
    for u, m, r in movie_ratings:
        if m not in m_c_map:
            m_c_map[m] = 0
        m_c_map[m] += 1

    top_movies = top_n(m_c_map, n=100)

    # Filter out the ratings that we don't need
    movie_ratings = [(uid, uri, rating) for uid, uri, rating in movie_ratings if uri in top_movies]

    # (Optional) count the number of movie ratings per user, filter them out as needed
    u_c_map = {}
    for u, m, r in movie_ratings:
        if u not in u_c_map:
            u_c_map[u] = 0
        u_c_map[u] += 1
    u_ids = [uid for uid, n_ratings in u_c_map.items() if n_ratings > 5]

    # Once again, filter the movie and entity ratings in case we lost some users there
    movie_ratings = [(uid, uri, rating) for uid, uri, rating in movie_ratings if uid in u_ids]

    # Now we only have the top-100 entities and movies, and we have the users we need.
    # Start mapping the users, movies and entities to indices
    uid_idx_map, uc = {}, 0
    m_uri_idx_map, mc = {}, 0

    for u, m, r in movie_ratings:
        if u not in uid_idx_map:
            uid_idx_map[u] = uc
            uc += 1
        if m not in m_uri_idx_map:
            m_uri_idx_map[m] = mc
            mc += 1


    # Convert the triples to index notation, convert ratings
    movie_ratings = [(uid_idx_map[u], m_uri_idx_map[m], r) for u, m, r in movie_ratings]

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
    [u.create_answer_eval_sets(train=True) for _, u in train_users.items()]
    [u.create_answer_eval_sets() for _, u in test_users.items()]

    # Done! Return the train and test users
    train_users = [u for _, u in train_users.items()]
    test_users = [u for _, u in test_users.items()]

    return train_users, test_users, uc, mc, 0


if __name__ == '__main__':
    # with open('../data/mindreader/entities.csv') as fp:
    #     entities = pd.read_csv(fp)
    # entity_map = {uri: name for uri, name in entities[['uri', 'name']].values}
    #
    # # 1. Functional Matrix Factorization on MindReader data, only asks towards movies
    # train_users, test_users, n_users, n_movies, n_entities, m_idx_uri_map, e_idx_uri_map = load_fmf_users(with_entities=False)
    # # print(len(train_users))
    # k = 10
    # m_idx_name_map = {}
    # e_idx_name_map = {}
    # for m in range(n_movies):
    #     m_idx_name_map[m] = entity_map[m_idx_uri_map[m]]
    # for e in range(n_entities):
    #     e_idx_name_map[e] = entity_map[e_idx_uri_map[e]]
    #
    # model = FunctionalMatrixFactorizaton(n_users, n_movies, n_entities, k=k, max_depth=3,
    #                                      entities_in_question_set=False,
    #                                      m_idx_name_map=m_idx_name_map,
    #                                      e_idx_name_map=e_idx_name_map)
    # model.fit(train_users, test_users)
    #
    # # 2. Functional Matrix Factorization on MindReader data, only asks towards entities
    # train_users, test_users, n_users, n_movies, n_entities, m_idx_uri_map, e_idx_uri_map = load_fmf_users(with_entities=True)
    # k = 10
    # m_idx_name_map = {}
    # e_idx_name_map = {}
    # for m in range(n_movies):
    #     m_idx_name_map[m] = entity_map[m_idx_uri_map[m]]
    # for e in range(n_entities):
    #     e_idx_name_map[e] = entity_map[e_idx_uri_map[e]]
    # model = FunctionalMatrixFactorizaton(n_users, n_movies, n_entities, k=k, max_depth=3,
    #                                      entities_in_question_set=True,
    #                                      m_idx_name_map=m_idx_name_map,
    #                                      e_idx_name_map=e_idx_name_map)
    # model.fit(train_users, test_users)

    # 3. Functional Matrix Factorization on MovieLens data
    train_users, test_users, n_users, n_movies, n_entities = load_fmf_users_movielens()
    k = 10
    model = FunctionalMatrixFactorizaton(n_users, n_movies, n_entities, k=k, max_depth=1,
                                         entities_in_question_set=False)
    model.fit(train_users, test_users)
