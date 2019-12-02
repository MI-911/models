from random import sample, choice

from networkx import pagerank_scipy, Graph

from data.graph_loader import load_graph
from data.training import cold_start
import numpy as np


def filter_map(u_r_map, condition):
    return {key: value for key, value in u_r_map.items() if condition(value)}


def filter_min_k(u_r_map, k):
    return filter_map(u_r_map, condition=lambda x: len(x['movies']) >= k and len(x['entities']) >= k)


def personalized_pagerank(G, movies=None, entities=None, alpha=0.85):
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


def precision_at_k(r, k):
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(ground_truth, prediction, k=10):
    r = get_relevance_list(ground_truth, prediction, k)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.sum(out) / min(k, len(ground_truth))


def get_relevance_list(ground_truth, prediction, k=10):
    return np.asarray([1 if item in ground_truth else 0 for item in prediction[:k]])


def hitrate(left_out, predicted, k=10):
    return 1 if left_out in predicted[:k] else 0


def construct_collaborative_graph(G, u_r_map, idx_movie, idx_entity):
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
        restrict_entities=None,
        split_ratio=[75, 25]
    )

    idx_entity = {value: key for key, value in entity_idx.items()}
    idx_movie = {value: key for key, value in movie_idx.items()}

    # G = load_graph(graph_path='../data/graph/triples.csv', directed=False)
    G = construct_collaborative_graph(Graph(), u_r_map, idx_movie, idx_entity)

    all_movies = set(movie_idx.keys())

    # Static, non-personalized measure of top movies
    top_movies = get_top_movies(u_r_map, idx_movie)

    # Try different samples
    for samples in range(1, 11):
        count = 0

        sum_popular_ap = 0
        sum_entity_ap = 0
        sum_movie_ap = 0

        hits_popular = 0
        hits_entity = 0
        hits_movie = 0

        for user, ratings in filter_min_k(u_r_map, samples).items():
            ground_truth = [idx_movie[head] for head, tail in ratings['test']]
            left_out = choice(ground_truth)

            if ground_truth:
                # Sample k entities
                sampled_entities = [idx_entity[head] for head, _ in sample(ratings['entities'], samples)]

                # Sample k movies
                sampled_movies = [idx_movie[head] for head, _ in sample(ratings['movies'], samples)]

                # Predict liked movies guessed on entities and movies
                entity_prediction = personalized_pagerank(G, all_movies, sampled_entities)
                movie_prediction = personalized_pagerank(G, all_movies, sampled_movies)

                # Get average precisions
                sum_popular_ap += average_precision(ground_truth, top_movies)
                sum_entity_ap += average_precision(ground_truth, entity_prediction)
                sum_movie_ap += average_precision(ground_truth, movie_prediction)

                # Get hit-rates
                hits_popular += hitrate(left_out, top_movies)
                hits_entity += hitrate(left_out, entity_prediction)
                hits_movie += hitrate(left_out, movie_prediction)

                count += 1

        print(f'{samples} samples:')
        print(f'Popular MAP: {sum_popular_ap / count}')
        print(f'Entity MAP: {sum_entity_ap / count}')
        print(f'Movie MAP: {sum_movie_ap / count}')
        print()

        print(f'Popular hit-rate: {hits_popular / count}')
        print(f'Entity hit-rate: {hits_entity / count}')
        print(f'Movie hit-rate: {hits_movie / count}')
        print()


if __name__ == '__main__':
    run()
