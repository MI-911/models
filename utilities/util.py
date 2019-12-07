from random import shuffle


def combine_movie_entity_index(data):
    combined = [{}, {}]

    index = 1
    for (u, m, rating, type) in data:
        if m not in combined[type]:
            combined[type][m] = index
            index += 1

    return combined


def filter_map(u_r_map, condition):
    return {key: value for key, value in u_r_map.items() if condition(value)}


def filter_min_k(u_r_map, k):
    return filter_map(u_r_map, condition=lambda x: len(x['movies']) >= k and len(x['entities']) >= k)


def get_top_movies(u_r_map, idx_entity):
    movie_count = {}

    for user, ratings in u_r_map.items():
        for movie, _ in ratings['movies']:
            movie_count[movie] = movie_count.get(movie, 0) + 1

    sorted_movies = sorted(list(movie_count.items()), key=lambda x: x[1], reverse=True)

    return [idx_entity[head] for head, tail in sorted_movies]


def get_entity_occurrence(u_r_map, idx_entity, idx_movie, include_test=False):
    subsets = {'movies': idx_movie, 'entities': idx_entity}

    if include_test:
        subsets['test'] = idx_movie

    entity_occurrence = {}
    for user, ratings in u_r_map.items():
        for subset, uri_resolver in subsets.items():
            for idx, _ in ratings[subset]:
                uri = uri_resolver[idx]
                entity_occurrence[uri] = entity_occurrence.get(uri, 0) + 1

    return entity_occurrence


def _prune(ratings, resolver, occurrence, keep):
    return [(idx, rating) for idx, rating in ratings if occurrence[resolver[idx]] >= keep]


def prune_low_occurrence(u_r_map, idx_entity, idx_movie, occurrence, keep=2, prune_test=False):
    subsets = {'movies': idx_movie, 'entities': idx_entity}

    if prune_test:
        subsets['test'] = idx_movie

    pruned_map = {}
    for user, ratings in u_r_map.items():
        pruned_map[user] = {
            subset: _prune(ratings[subset], resolver, occurrence, keep) for subset, resolver in subsets.items()
        }

        if not prune_test:
            pruned_map[user]['test'] = ratings['test']

    return pruned_map


def split_users(all_users, train_ratio):
    shuffle(all_users)
    split_index = int(len(all_users) * train_ratio)

    return all_users[:split_index], all_users[split_index:]