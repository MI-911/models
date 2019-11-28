from csv import DictReader

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
