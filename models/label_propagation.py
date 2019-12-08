import networkx as nx
from data.graph_loader import load_graph
from data.training import load_label_propagation_warm_start
import numpy as np
import matplotlib.pyplot as plt

def build_graph():
    G = load_graph(graph_path='../data/graph/triples.csv')

    nx.set_node_attributes(G, {n: False for n in G.nodes()}, 'labelled')

    return G


def initialize_scores(G, likes, dislikes):
    nx.set_node_attributes(G, {
        n: 1 if n in likes else 0 if n in dislikes else 0
        for n in G.nodes()
    }, 'score')

    nx.set_node_attributes(G, {
        n: True if n in likes or n in dislikes else False
        for n in G.nodes()
    }, 'labelled')


def calculate_avg_score(G, node):
    score_sum = 0
    n_neighbors = 0
    for neighbor_id in G.neighbors(node):
        score_sum += G._node[neighbor_id]['score']
        n_neighbors += 1
    return score_sum / n_neighbors


def propagate(G):
    next_scores = {}
    for node, attrs in G.nodes.data():
        if attrs['labelled']:
            # Scores of labeled nodes do not change
            next_scores[node] = attrs['score']
        else:
            next_scores[node] = calculate_avg_score(G, node)

    nx.set_node_attributes(G, next_scores, 'score')


def predict(G):
    score_dict = nx.get_node_attributes(G, 'score')
    sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    return [node for node, score in sorted_scores if not score == 1 and not score == -1]


def average_precision(user, predictions):
    print(u.test_likes)
    print(predictions)
    print()
    n = len(predictions)
    n_correct = 0
    pre_avg = 0

    for _i, node in enumerate(predictions):
        if node in user.test_likes:
            n_correct += 1
            pre_avg += n_correct / _i

    return pre_avg / n


if __name__ == '__main__':
    KG = build_graph()
    users = load_label_propagation_warm_start()

    average_precisions = []

    for u in users:
        u.split()

        initialize_scores(KG, u.likes, u.dislikes)

        pos = nx.spring_layout(KG)
        nx.draw_networkx_nodes(KG, pos)
        nx.draw_networkx_edges(KG, pos)
        plt.show()

        for i in range(10):
            propagate(KG)

        ranked_nodes = predict(KG)
        ap = average_precision(u, ranked_nodes[:20])
        print(f'Average precision is {ap}')
        average_precisions.append(ap)  # AP@20

    print(f'MAP@20 is {np.mean(average_precisions)}')




