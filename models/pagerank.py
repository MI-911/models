from collections import defaultdict
from random import sample, choice

from networkx import pagerank_scipy, Graph

from data.graph_loader import load_graph
from data.training import cold_start
from utilities.metrics import average_precision, hitrate, ndcg_at_k
from utilities.util import get_top_movies, filter_min_k, get_entity_occurrence, prune_low_occurrence


def personalized_pagerank(G, movies=None, entities=None, alpha=0.85):
    if not movies:
        return []

    personalization = {entity: 1 for entity in entities} if entities else None

    res = list(pagerank_scipy(G, alpha=alpha, personalization=personalization).items())

    # Filter movies only and return sorted list
    filtered = [(head, tail) for head, tail in res if head in movies and head not in entities]
    return [head for head, _ in sorted(filtered, key=lambda pair: pair[1], reverse=True)]


def remove_nodes(G, keep):
    all_nodes = list(G.nodes)

    for n in all_nodes:
        if n not in keep:
            G.remove_node(n)


def construct_collaborative_graph(G, u_r_map, idx_movie, idx_entity, exclude=None):
    for user, ratings in u_r_map.items():
        G.add_node(user)

        for head, rating in ratings['movies']:
            head = idx_movie[head]
            if exclude and (user, head) in exclude:
                continue

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
        split_ratio=[100, 0]
    )

    idx_entity = {value: key for key, value in entity_idx.items()}
    idx_movie = {value: key for key, value in movie_idx.items()}

    all_movies = set(movie_idx.keys())

    # Get entity frequency
    entity_frequency = defaultdict(int)
    for user, ratings in u_r_map.items():
        for idx, rating in ratings['movies']:
            uri = idx_movie[idx]
            entity_frequency[uri] += 1

        for idx, rating in ratings['entities']:
            uri = idx_entity[idx]
            entity_frequency[uri] += 1

    # Filter entities with only one occurrence
    entity_occurrence = get_entity_occurrence(u_r_map, idx_entity, idx_movie)
    u_r_map = prune_low_occurrence(u_r_map, idx_entity, idx_movie, entity_occurrence)

    # Static, non-personalized measure of top movies
    top_movies = get_top_movies(u_r_map, idx_movie)

    G = load_graph(graph_path='../data/graph/triples.csv', directed=False)

    # Try different samples
    k = 10
    filtered = filter_min_k(u_r_map, 5).items()
    for samples in range(1, 6):
        count = 0

        sum_popular_ap = 0
        sum_entity_ap = 0
        sum_movie_ap = 0

        hits_popular = 0
        hits_entity = 0
        hits_movie = 0

        sum_popular_ndcg = 0
        sum_entity_ndcg = 0
        sum_movie_ndcg = 0

        for user, ratings in filtered:
            # Sample k entities
            sampled_entities = [idx_entity[head] for head, _ in sample(ratings['entities'], samples)]

            # Sample k movies
            sampled_movies = [idx_movie[head] for head, _ in sample(ratings['movies'], samples)]

            # Use remaining movies as ground truth
            ground_truth = [idx_movie[head] for head, tail in ratings['movies'] if head not in sampled_movies][:1]
            left_out = choice(ground_truth)

            if ground_truth:
                # Predict liked movies guessed on entities and movies
                entity_prediction = personalized_pagerank(G, all_movies, sampled_entities)
                movie_prediction = personalized_pagerank(G, all_movies, sampled_movies)

                # Get average precisions
                sum_popular_ap += average_precision(ground_truth, top_movies, k)
                sum_entity_ap += average_precision(ground_truth, entity_prediction, k)
                sum_movie_ap += average_precision(ground_truth, movie_prediction, k)

                # Get hit-rates
                hits_popular += hitrate(left_out, top_movies, k)
                hits_entity += hitrate(left_out, entity_prediction, k)
                hits_movie += hitrate(left_out, movie_prediction, k)

                # Get NDCGs
                sum_popular_ndcg += ndcg_at_k(ground_truth, top_movies, k)
                sum_entity_ndcg += ndcg_at_k(ground_truth, entity_prediction, k)
                sum_movie_ndcg += ndcg_at_k(ground_truth, movie_prediction, k)

                count += 1

        print(f'{samples} samples:')
        print(f'Popular MAP: {sum_popular_ap / count * 100}%')
        print(f'Entity MAP: {sum_entity_ap / count * 100}%')
        print(f'Movie MAP: {sum_movie_ap / count * 100}%')
        print()

        print(f'Popular NDCG: {sum_popular_ndcg / count * 100}%')
        print(f'Entity NDCG: {sum_entity_ndcg / count * 100}%')
        print(f'Movie NDCG: {sum_movie_ndcg / count * 100}%')
        print()

        print(f'Popular hit-rate: {hits_popular / count * 100}%')
        print(f'Entity hit-rate: {hits_entity / count * 100}%')
        print(f'Movie hit-rate: {hits_movie / count * 100}%')
        print()


if __name__ == '__main__':
    run()
