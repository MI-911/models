import json
import pandas as pd
from random import shuffle


def _add_rating_w_user(lst, u, m, rating, conversion_map, is_movie=False):
    if conversion_map:
        assert rating in conversion_map, f'Rating value {rating} not found in conversion map {conversion_map}'
        rating = conversion_map[rating]
        if not rating:
            return  # Don't add the rating if the value is None or False
        else:
            lst.append((u, m, rating, 1 if is_movie else 0))


def _add_rating(lst, m, rating, conversion_map):
    if conversion_map:
        assert rating in conversion_map, f'Rating value {rating} not found in conversion map {conversion_map}'
        rating = conversion_map[rating]
        if not rating:
            return  False # Don't add the rating if the value is None or False

    lst.append((m, rating))


def warm_start(
        ratings_path='../data/mindreader/ratings_clean.json',
        entities_path='../data/mindreader/entities_clean.json',
        conversion_map=None,
        split_ratio=[75, 25]):
    """
    Converts UIDs and URIs to indices as returns the ratings
    in a list fashion (note different indices from movies
    and entities.).
    :param ratings_path: Where to load ratings from.
    :param entities_path: Where to load entities from.
    :param conversion_map: A dictionary to map rating values to different values, e.g.
           {
                -1 : 1,
                0 : None  // This ignores the rating, won't be included in the ratings map.
                1 : 5
           }
    :param split_ratio: The ratio between the size of the training and test set of MOVIE ratings.
    :return: Train set, test set, n_users, n_movies, n_entities
    """

    assert sum(split_ratio) == 100 or sum(split_ratio) == 1, f'Split ratios {split_ratio} does not add up to 1 or 100.'

    u_uid_map, uc = {}, 0
    m_uri_map, mc = {}, 0
    e_uri_map, ec = {}, 0

    with open(ratings_path) as fp:
        ratings = json.load(fp)
    with open(entities_path) as fp:
        entities = json.load(fp)

    # Map URIs to labels
    label_map = {}
    for uri, name, labels in entities:
        labels = labels.split('|')
        if uri not in label_map:
            label_map[uri] = labels

    # Split into movie and entity ratings
    movie_ratings = [(uid, uri, r) for uid, uri, r in ratings if 'Movie' in label_map[uri]]
    entity_ratings = [(uid, uri, r) for uid, uri, r in ratings if 'Movie' not in label_map[uri]]

    for uid, uri, rating in movie_ratings:
        if uid not in u_uid_map:
            u_uid_map[uid] = uc
            uc += 1
        if uri not in m_uri_map:
            m_uri_map[uri] = mc
            mc += 1
    for uid, uri, rating in entity_ratings:
        if uid not in u_uid_map:
            u_uid_map[uid] = uc
            uc += 1
        if uri not in e_uri_map:
            e_uri_map[uri] = ec
            ec += 1

    _m_ratings = []
    _e_ratings = []
    for uid, uri, rating in movie_ratings:
        _add_rating_w_user(_m_ratings, u_uid_map[uid], m_uri_map[uri], rating, conversion_map, is_movie=True)

    for uid, uri, rating in entity_ratings:
        _add_rating_w_user(_e_ratings, u_uid_map[uid], e_uri_map[uri], rating, conversion_map, is_movie=False)

    movie_ratings = _m_ratings
    entity_ratings = _e_ratings

    shuffle(movie_ratings)
    shuffle(entity_ratings)

    m_split_idx = int(len(movie_ratings) * (split_ratio[0] / 100))
    train_movies = movie_ratings[:m_split_idx]
    test_movies = movie_ratings[m_split_idx:]

    train = entity_ratings + train_movies
    test = test_movies

    shuffle(train)
    shuffle(test)

    return train, test, uc, mc, ec


def cold_start(from_path='../data/user_ratings_map.json', conversion_map=None, split_ratio=[75, 25]):
    """
    Converts UIDs and URIs to indices as returns the ratings
    in a user-major fashion (note different indices from movies
    and entities.).

    :param from_path: Where to load user-major ratings from.
    :param conversion_map: A dictionary to map rating values to different values, e.g.
           {
                -1 : 1,
                0 : None  // This ignores the rating, won't be included in the ratings map.
                1 : 5
           }
    :param split_ratio: The ratio between the size of the training and test set of MOVIE ratings.
    :return: Indexed user-major ratings, n_users, n_movies, n_entities
    """

    assert sum(split_ratio) == 100 or sum(split_ratio) == 1, f'Split ratios {split_ratio} does not add up to 1 or 100.'

    u_idx_map, uc = {}, 0
    m_uri_map, mc = {}, 0
    e_uri_map, ec = {}, 0

    with open(from_path) as fp:
        u_r_map = json.load(fp)

    idx_u_r_map = {}

    for u, ratings in u_r_map.items():
        if u not in u_idx_map:
            u_idx_map[u] = uc
            uc += 1
        for m, rating in ratings['movies']:
            if m not in m_uri_map:
                m_uri_map[m] = mc
                mc += 1
        for e, rating in ratings['entities']:
            if e not in e_uri_map:
                e_uri_map[e] = ec
                ec += 1

        u = u_idx_map[u]
        if u not in idx_u_r_map:
            idx_u_r_map[u] = {'movies': [], 'entities': []}

        # Add movie ratings
        for m, rating in ratings['movies']:
            m = m_uri_map[m]
            _add_rating(idx_u_r_map[u]['movies'], m, rating, conversion_map)

        # Add entity ratings
        for e, rating in ratings['entities']:
            e = e_uri_map[e]
            _add_rating(idx_u_r_map[u]['entities'], e, rating, conversion_map)

    # Split movie ratings into training and test sets
    for u, ratings in idx_u_r_map.items():
        m_ratings = idx_u_r_map[u]['movies']
        split_index = int(len(m_ratings) * (split_ratio[0] / 100))

        shuffle(m_ratings)
        training = m_ratings[split_index:]
        test = m_ratings[:split_index]

        idx_u_r_map[u]['movies'] = training
        idx_u_r_map[u]['test'] = test

    return idx_u_r_map, len(u_idx_map), len(m_uri_map), len(e_uri_map)


if __name__ == '__main__':
    data = warm_start(
        ratings_path='../data/mindreader/ratings_clean.json',
        entities_path='../data/mindreader/entities_clean.json',
        conversion_map={
            -1: 1,
            0: None,
            1: 5
        },
        split_ratio=[80, 20]
    )

    print(data)
