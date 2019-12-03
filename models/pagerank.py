from random import sample, choice

from networkx import pagerank_scipy, Graph

from data.graph_loader import load_graph
from data.training import cold_start
import numpy as np

from utilities.metrics import average_precision, hitrate
from utilities.util import get_top_movies, filter_min_k


def personalized_pagerank(G, movies=None, entities=None, alpha=0.75):
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
        restrict_entities=['Person'],
        split_ratio=[100, 0]
    )

    idx_entity = {value: key for key, value in entity_idx.items()}
    idx_movie = {value: key for key, value in movie_idx.items()}

    all_movies = set(movie_idx.keys())

    # Get entity frequency
    entity_frequency = {}
    for user, ratings in u_r_map.items():
        for idx, rating in ratings['movies']:
            uri = idx_movie[idx]
            entity_frequency[uri] = entity_frequency.get(uri, 0) + 1

        for idx, rating in ratings['entities']:
            uri = idx_entity[idx]
            entity_frequency[uri] = entity_frequency.get(uri, 0) + 1

    # Remove entities that are only observed once
    for user, ratings in u_r_map.items():
        new_movies, new_entities = [], []

        for idx, rating in ratings['movies']:
            if entity_frequency[idx_movie[idx]] > 1:
                new_movies.append((idx, rating))
            else:
                print(f'Bye {idx_movie[idx]}')

        for idx, rating in ratings['entities']:
            if entity_frequency[idx_entity[idx]] > 1:
                new_entities.append((idx, rating))
            else:
                print(f'Bye {idx_entity[idx]}')

        u_r_map[user] = {'movies': new_movies, 'entities': new_entities}

    # Static, non-personalized measure of top movies
    top_movies = get_top_movies(u_r_map, idx_movie)

    # Try different samples
    k = 10
    for samples in range(1, 11):
        count = 0

        sum_popular_ap = 0
        sum_entity_ap = 0
        sum_movie_ap = 0

        hits_popular = 0
        hits_entity = 0
        hits_movie = 0

        for user, ratings in filter_min_k(u_r_map, samples).items():
            # Sample k entities
            sampled_entities = [idx_entity[head] for head, _ in sample(ratings['entities'], samples)]

            # Sample k movies
            sampled_movies = [idx_movie[head] for head, _ in sample(ratings['movies'], samples)]

            # Use remaining movies as ground truth
            ground_truth = [idx_movie[head] for head, tail in ratings['movies'] if head not in sampled_movies][:1]
            left_out = choice(ground_truth)

            # Construct graph without user's ground truth ratings
            exclude_ratings = {(user, movie) for movie in ground_truth}
            G = construct_collaborative_graph(Graph(), u_r_map, idx_movie, idx_entity, exclude=exclude_ratings)

            if ground_truth:
                # Predict liked movies guessed on entities and movies
                entity_prediction = personalized_pagerank(G, all_movies, sampled_entities)
                # movie_prediction = personalized_pagerank(G, all_movies, sampled_movies)

                # Get average precisions
                sum_popular_ap += average_precision(ground_truth, top_movies, k)
                sum_entity_ap += average_precision(ground_truth, entity_prediction, k)
                # sum_movie_ap += average_precision(ground_truth, movie_prediction)

                # Get hit-rates
                hits_popular += hitrate(left_out, top_movies, k)
                hits_entity += hitrate(left_out, entity_prediction, k)
                # hits_movie += hitrate(left_out, movie_prediction)

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
