import pandas as pd
import numpy as np
import torch

from tqdm import tqdm


def load_adjacency_matrix(path):
    df = pd.read_csv(path)[['head_uri', 'relation', 'tail_uri']]

    entities = set(df['head_uri']).union(set(df['tail_uri']))
    entitity_index = {uri: i+1 for i, uri in enumerate(entities)}
    num_entities = len(entities)+1

    relations = set(df['relation'])
    relation_index = {r: i for i, r in enumerate(relations)}

    values = np.zeros((len(df)*2,))
    indices = np.zeros((len(df)*2, 2))
    for index, (head, relation, tail) in tqdm(df[:5000].iterrows(), desc='Creating matrix', total=len(df)):
        cur_index = index * 2
        h_index = entitity_index[head]
        t_index = entitity_index[tail]
        r_index = relation_index[relation]

        values[cur_index:cur_index+2] = [r_index]*2
        indices[cur_index:cur_index + 2, :] = [[h_index, t_index], [t_index, h_index]]

    values = torch.LongTensor(values)
    indices = torch.LongTensor(indices).t()

    # adjacency_matrix = torch.sparse.LongTensor(indices, values, torch.Size([num_entities, num_entities]))

    return values, indices, entitity_index, relation_index, num_entities


if __name__ == '__main__':
    a = load_adjacency_matrix('mindreader/triples.csv')
    print('Hej')