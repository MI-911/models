import json
import pandas as pd
from random import shuffle


def _add_rating(lst, m, rating, conversion_map):
    if conversion_map:
        assert rating in conversion_map, f'Rating value {rating} not found in conversion map {conversion_map}'
        rating = conversion_map[rating]
        if not rating:
            return  # Don't add the rating if the value is None or False

    lst.append((m, rating))


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
        for m, rating in ratings['movies']:
            m = m_uri_map[m]
            _add_rating(idx_u_r_map[u]['movies'], m, rating, conversion_map)
        for e, rating in ratings['entities']:
            e = e_uri_map[e]
            _add_rating(idx_u_r_map[u]['movies'], e, rating, conversion_map)

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
    data = cold_start(
        from_path='../data/mindreader/user_ratings_map.json',
        conversion_map={
            -1: 1,
            0: None,
            1: 5
        },
        split_ratio=[80, 20]
    )

    print(data)
