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
