from csv import DictReader
import json
import networkx
from itertools import combinations
import numpy as np

def load_graph(graph_path, directed=True, exclude_relations=None, restrict_nodes=None):
    G = networkx.nx.DiGraph() if directed else networkx.nx.Graph()

    with open(graph_path, 'r') as graph_fp:
        graph_reader = DictReader(graph_fp)

        for row in graph_reader:
            h, r, t = row['head_uri'], row['relation'], row['tail_uri']
            if exclude_relations and r in exclude_relations:
                continue
            if restrict_nodes and h in restrict_nodes and t in restrict_nodes:
                G.add_node(row['head_uri'])
                G.add_node(row['tail_uri'])
                G.add_edge(row['head_uri'], row['tail_uri'], type=row['relation'])

    return G


def get_maps(entities_path):
    with open(entities_path) as fp:
        entities = json.load(fp)

    # Map URIs to labels, names
    label_map = {}
    name_map = {}
    for uri, name, labels in entities:
        labels = labels.split('|')
        if uri not in label_map:
            label_map[uri] = labels
        if uri not in name_map:
            name_map[uri] = name

    return label_map, name_map


def load_labelled_graph(graph_path, directed=True, exclude_relations=None, restrict_nodes=None):
    label_map, name_map = get_maps('../data/mindreader/entities_clean.json')
    G = load_graph(graph_path, directed, exclude_relations, restrict_nodes)

    networkx.set_node_attributes(G, label_map, 'labels')
    networkx.set_node_attributes(G, name_map, 'name')

    # print(networkx.get_node_attributes(G, 'labels'))

    return G


class CollaborativeKnowledgeGraph:
    def __init__(self, KG):
        self.KG = KG
        self.precomputed_rankings = {}

    def ppr_top_n(self, source_nodes, top_n=20):
        source_node_encoding = ''.join(map(str, sorted(source_nodes)))
        if source_node_encoding in self.precomputed_rankings:
            ranked_movies = self.precomputed_rankings[source_node_encoding]
        else:
            # Set the weight to 1 for all source nodes
            # TODO: Consider setting the weight depending on popularity or contentiousness
            source_nodes = {n: 1 for n in source_nodes}

            ranked_movies = networkx.pagerank(self.KG, personalization=source_nodes)
            ranked_movies = list(sorted(ranked_movies.items(), key=lambda x: x[1], reverse=True))
            ranked_movies = [m for m, c in ranked_movies if isinstance(m, int)]  # Movies are integers, users are strings
            self.precomputed_rankings[source_node_encoding] = ranked_movies

        return ranked_movies[:top_n]

    @staticmethod
    def load_from_users(user_set, directed=False, liked_value=1, disliked_value=-1):
        # Since user and movie indices are both integers in the same space,
        # we denote users with string integers and movies with std. integers.
        KG = networkx.DiGraph() if directed else networkx.Graph()
        for u in user_set:
            u_id = str(u.idx)
            KG.add_node(u_id)
            for m, r in u.movie_ratings:
                # TODO: We are only adding LIKED edges now. Consider
                #       doing something different different
                if r == liked_value:
                    KG.add_edge(u_id, m)

        return CollaborativeKnowledgeGraph(KG)


class KnowledgeGraph:
    def __init__(self, KG, ):
        self.KG = KG
        self.movie_uris = [
            uri
            for uri, labels
            in networkx.get_node_attributes(self.KG, 'labels').items()
            if 'Movie' in labels
        ]
        self.precomputed_rankings = {}

    @staticmethod
    def load_from_triples(graph_path, directed=True, exclude_relations=None, restrict_nodes=None):
        return KnowledgeGraph(load_labelled_graph(graph_path, directed, exclude_relations, restrict_nodes))

    def ppr_top_n(self, source_nodes, top_n=20):
        source_nodes = list(source_nodes)
        source_node_encoding = ''.join(map(str, sorted(source_nodes)))
        if source_node_encoding in self.precomputed_rankings:
            sorted_ranked_movie_nodes = self.precomputed_rankings[source_node_encoding]

        else:
            # Set the weight to 1 for all source nodes
            # TODO: Consider setting the weight depending on popularity or contentiousness
            source_nodes = {n: 1 for n in source_nodes}
            ranked_nodes = networkx.pagerank(self.KG, personalization=source_nodes)

            ranked_movie_nodes = [
                (node, pr)
                for node, pr
                in ranked_nodes.items()
                if node in self.movie_uris and node not in source_nodes
            ]
            sorted_ranked_movie_nodes = list(sorted(ranked_movie_nodes, key=lambda x: x[1], reverse=True))
            self.precomputed_rankings[source_node_encoding] = sorted_ranked_movie_nodes

        return sorted_ranked_movie_nodes[:top_n]


if __name__ == '__main__':
    load_labelled_graph('../graph/triples.csv')
