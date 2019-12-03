from csv import DictReader
import json
import networkx


def load_graph(graph_path, directed=True, exclude_relations=None):
    G = networkx.nx.DiGraph() if directed else networkx.nx.Graph()

    with open(graph_path, 'r') as graph_fp:
        graph_reader = DictReader(graph_fp)

        for row in graph_reader:
            if exclude_relations and row['relation'] in exclude_relations:
                continue
            
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


def load_labelled_graph(graph_path, directed=True, exclude_relations=None):
    label_map, name_map = get_maps('mindreader/entities_clean.json')
    G = load_graph(graph_path, directed, exclude_relations)

    networkx.set_node_attributes(G, label_map, 'labels')
    networkx.set_node_attributes(G, name_map, 'name')

    # print(networkx.get_node_attributes(G, 'labels'))

    return G


if __name__ == '__main__':
    load_labelled_graph('graph/triples.csv')
