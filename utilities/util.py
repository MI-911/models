def combine_movie_entity_index(data):
    combined = [{}, {}]

    index = 1
    for (u, m, rating, type) in data:
        if m not in combined[type]:
            combined[type][m] = index
            index += 1

    return combined
