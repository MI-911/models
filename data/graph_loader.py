from csv import DictReader
import json
import networkx


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


class KnowledgeGraph:
    def __init__(self, KG, ):
        self.KG = KG
        self.movie_uris = [
            uri
            for uri, labels
            in networkx.get_node_attributes(self.KG, 'labels').items()
            if 'Movie' in labels
        ]

    @staticmethod
    def load_from_triples(graph_path, directed=True, exclude_relations=None, restrict_nodes=None):
        return KnowledgeGraph(load_labelled_graph(graph_path, directed, exclude_relations, restrict_nodes))

    def ppr_top_n(self, source_nodes, top_n=20):
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
        sorted_ranked_movie_nodes = list(sorted(ranked_nodes.items(), key=lambda x: x[1], reverse=True))
        if len(sorted_ranked_movie_nodes) == 0:
            print('asd')
        return sorted_ranked_movie_nodes[:top_n]


if __name__ == '__main__':
    load_labelled_graph('../graph/triples.csv')
