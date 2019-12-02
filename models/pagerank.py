from random import sample, choice

from networkx import pagerank_scipy, Graph

from data.training import cold_start
import numpy as np


def filter_map(u_r_map, condition):
    return {key: value for key, value in u_r_map.items() if condition(value)}


def filter_min_k(u_r_map, k):
    return filter_map(u_r_map, condition=lambda x: len(x['movies']) >= k and len(x['entities']) >= k)


def personalized_pagerank(G, movies=None, entities=None, alpha=0.8):
    if not movies:
        return []

    personalization = {entity: 1 for entity in entities} if entities else None

    res = list(pagerank_scipy(G, alpha=alpha, personalization=personalization).items())

    # Filter movies only and return sorted list
    filtered = [(head, tail) for head, tail in res if head in movies and head not in entities]
    return [head for head, _ in sorted(filtered, key=lambda pair: pair[1], reverse=True)]


def get_top_movies(u_r_map, idx_entity):
    movie_count = {}

    for user, ratings in u_r_map.items():
        for movie, _ in ratings['movies']:
            movie_count[movie] = movie_count.get(movie, 0) + 1

    sorted_movies = sorted(list(movie_count.items()), key=lambda x: x[1], reverse=True)

    return [idx_entity[head] for head, tail in sorted_movies]


def remove_nodes(G, keep):
    all_nodes = list(G.nodes)

    for n in all_nodes:
        if n not in keep:
            G.remove_node(n)


def precision_at_k(ground_truth, prediction, r, k):
    prediction = np.asarray([1 if item in ground_truth[:k] else 0 for item in prediction[:k]])

    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(ground_truth, predicted, k=5):
    p_sum = 0

    unseen = list(ground_truth)
    seen = list()

    for i in range(0, k):
        at_i = predicted[i]

        if at_i in unseen:
            unseen.remove(at_i)
            seen.append(at_i)

            p_sum += len(seen) / (i + 1)

    return p_sum / len(ground_truth)


def hitrate(ground_truth, predicted, k=5):
    sampled_item = choice(ground_truth)

    return 1 if sampled_item in predicted[:k] else 0


def construct_collaborative_graph(u_r_map, idx_movie, idx_entity):
    G = Graph()
    for user, ratings in u_r_map.items():
        G.add_node(user)

        for head, rating in ratings['movies']:
            head = idx_movie[head]
            G.add_node(head)
            G.add_edge(user, head)

        for head, rating in ratings['entities']:
            head = idx_entity[head]
            G.add_node(head)
            G.add_edge(user, head)

    return G


def run():
    u_r_map, n_users, movie_idx, entity_idx = cold_start(
        from_path='../data/mindreader/user_ratings_map.json',
        conversion_map={
            -1: None,
            0: None,  # Ignore don't know ratings
            1: 1
        },
        restrict_entities=['Category'],
        split_ratio=[75, 25]
    )

    idx_entity = {value: key for key, value in entity_idx.items()}
    idx_movie = {value: key for key, value in movie_idx.items()}

    print(idx_entity[0])
    print(idx_movie[0])

    for k in range(1, 10):
        filtered = filter_min_k(u_r_map, k)

        print(f'{k}: {len(filtered)}')

    # G = load_graph('../data/graph/triples.csv', directed=False)
    G = construct_collaborative_graph(u_r_map, idx_movie, idx_entity)

    all_movies = set(movie_idx.keys())

    # Static, non-personalized measure of top movies
    top_movies = get_top_movies(u_r_map, idx_movie)

    for samples in range(1, 11):
        count = 0

        sum_popular_ap = 0
        sum_entity_ap = 0
        sum_movie_ap = 0
        sum_global_ap = 0

        hits_popular = 0
        hits_entity = 0
        hits_movie = 0
        hits_global = 0

        for user, ratings in filter_min_k(u_r_map, samples).items():
            movies = [idx_movie[head] for head, tail in ratings['movies']]

            if movies:
                # Sample k entities
                sampled_entities = [idx_entity[head] for head, _ in sample(ratings['entities'], samples)]

                # Sample k movies
                sampled_movies = [idx_movie[head] for head, _ in sample(ratings['movies'], samples)]

                # Predicted liked movies guessed on entities and movies
                entity_prediction = personalized_pagerank(G, all_movies, sampled_entities)
                movie_prediction = personalized_pagerank(G, all_movies, sampled_movies)

                # Predicted like movies guessed on global PageRank
                global_prediction = personalized_pagerank(G, all_movies, [])

                # Get average precisions
                sum_popular_ap += average_precision(movies, top_movies)
                sum_entity_ap += average_precision(movies, entity_prediction)
                sum_movie_ap += average_precision(movies, movie_prediction)
                sum_global_ap += average_precision(movies, global_prediction)

                # Get hit-rates
                hits_popular += hitrate(movies, top_movies)
                hits_entity += hitrate(movies, entity_prediction)
                hits_movie += hitrate(movies, movie_prediction)
                hits_global += hitrate(movies, global_prediction)

                count += 1

        print(f'{samples} samples:')
        print(f'Popular MAP: {sum_popular_ap / count}')
        print(f'Entity MAP: {sum_entity_ap / count}')
        print(f'Movie MAP: {sum_movie_ap / count}')
        print(f'Global MAP: {sum_global_ap / count}')
        print()

        print(f'Popular hit-rate: {hits_popular / count}')
        print(f'Entity hit-rate: {hits_entity / count}')
        print(f'Movie hit-rate: {hits_movie / count}')
        print(f'Global hit-rate: {hits_global / count}')
        print()

    print(G)


if __name__ == '__main__':
    run()
