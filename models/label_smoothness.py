import numpy as np
import torch
from torch import nn


class GNN(nn.Module):
    def __init__(self, n_users, n_relations, n_entities, values, indices, latent_dim=8, batch_size=1, cuda=False):
        super(GNN, self).__init__()

        # Vars
        self.n_users = n_users
        self.n_relations = n_relations
        self.n_entities = n_entities
        self.latent_dim = latent_dim
        self.batch_size = batch_size

        self.user_emb = nn.Embedding(n_users, latent_dim)
        self.relation_emb = nn.Parameter(torch.rand((n_relations, latent_dim)), requires_grad=True).expand(batch_size, n_relations, latent_dim)
        self.entity_emb = nn.Parameter(torch.rand((n_entities, latent_dim)), requires_grad=True)
        self.adjacency_matrix = torch.sparse.FloatTensor(indices, values, torch.Size([n_entities, n_entities])).\
            to_dense()
        self.size = torch.Size([n_entities, n_entities])

        # GNN layer 0
        self.weight_0 = nn.Parameter(torch.rand((latent_dim, latent_dim)), requires_grad=True)
        self.gnn_last_layer_activation = nn.Tanh()

        # Prediction
        self.pred_activation = nn.Sigmoid()

    def forward(self, user, entity):
        u_emb = self.user_emb(user)
        r = nn.Embedding(self.n_entities, 1, padding_idx=0)
        b = torch.einsum('bi,bji->bj', u_emb, self.relation_emb)

        user_a_m = torch.FloatTensor(np.zeros((self.batch_size, self.n_entities, self.n_entities)))

        for i in range(self.batch_size):
            r.weights = b[i]
            user_a_m[i] = r(self.adjacency_matrix).squeeze(2)

        # Layer 0
        a = user_a_m.matmul(self.entity_emb)
        l0 = a.matmul(self.weight_0)

        last_layer = l0[torch.arange(self.batch_size), entity]
        user_entity_representation = self.gnn_last_layer_activation(last_layer)

        # Pred
        p = torch.einsum('bi,bi->b', u_emb, user_entity_representation)

        return p





