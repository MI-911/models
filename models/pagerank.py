from data.training import cold_start
from data.graph_loader import load_graph
from networkx import pagerank_numpy, pagerank, pagerank_scipy


def filter_map(u_r_map, condition):
    return {key: value for key, value in u_r_map.items() if condition(value)}


def filter_min_k(u_r_map, k):
    return filter_map(u_r_map, condition=lambda x: len(x['movies']) >= k and len(x['entities']) >= k)


def personalized_pagerank(G, movies=None, entities=None, top_k=10, alpha=0.85):
    if not movies:
        return []

    personalization = {entity: 1 / len(entities) for entity in entities} if entities else None

    res = list(pagerank_numpy(G, alpha=0.85, personalization=personalization).items())

    # Filter movies only and return sorted list
    filtered = [(head, tail) for head, tail in res if head in movies and head not in entities]
    return sorted(filtered, key=lambda pair: pair[1], reverse=True)[:10]


def remove_nodes(G, keep):
    all_nodes = list(G.nodes)

    for n in all_nodes:
        if n not in keep:
            G.remove_node(n)


def run():
    u_r_map, n_users, movie_idx, entity_idx = cold_start(
        from_path='../data/mindreader/user_ratings_map.json',
        conversion_map={
            -1: None,
            0: None,  # Ignore don't know ratings
            1: 1
        },
        split_ratio=[75, 25]
    )

    idx_entity = {value: key for key, value in entity_idx.items()}
    idx_movie = {value: key for key, value in movie_idx.items()}

    print(idx_entity[0])
    print(idx_movie[0])

    for k in range(1, 10):
        filtered = filter_min_k(u_r_map, k)

        print(f'{k}: {len(filtered)}')

    G = load_graph('../data/graph/triples.csv', directed=False)

    movies = set(movie_idx.keys())
    print(personalized_pagerank(G, movies, ['http://www.wikidata.org/entity/Q134773']))
    print(personalized_pagerank(G, movies, ['http://www.wikidata.org/entity/Q134773']))
    print(personalized_pagerank(G, movies, ['http://www.wikidata.org/entity/Q134773']))
    print(personalized_pagerank(G, movies, ['http://www.wikidata.org/entity/Q134773']))
    print(personalized_pagerank(G, movies, ['http://www.wikidata.org/entity/Q134773']))
    print(personalized_pagerank(G, movies, ['http://www.wikidata.org/entity/Q134773']))
    print(personalized_pagerank(G, movies, ['http://www.wikidata.org/entity/Q134773']))
    print(personalized_pagerank(G, movies, ['http://www.wikidata.org/entity/Q134773']))
    print(personalized_pagerank(G, movies, ['http://www.wikidata.org/entity/Q134773']))

    print(G)


if __name__ == '__main__':
    run()
