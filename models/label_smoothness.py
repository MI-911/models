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
        self.values = values
        self.indices = indices
        self.size = torch.Size([n_entities, n_entities])

        # GNN layer 0
        self.weight_0 = nn.Parameter(torch.rand((latent_dim, latent_dim)), requires_grad=True)
        self.gnn_last_layer_activation = nn.Tanh()

        # Prediction
        self.pred_activation = nn.Sigmoid()

    def forward(self, user, entity):
        u_emb = self.user_emb(user)
        relation_weights = torch.einsum('bi,bji->bj', u_emb, self.relation_emb)

        v = torch.FloatTensor(np.zeros((self.batch_size, len(self.values))))

        for i, value in enumerate(self.values):
            v[:, i] = relation_weights[:, value]

        adjencency_matrix = torch.FloatTensor(np.zeros((self.batch_size, self.n_entities, self.n_entities)))
        for i in range(self.batch_size):
            adjencency_matrix[i] = torch.sparse.FloatTensor(self.indices, v[i], self.size).to_dense()

        # Layer 0
        l0 = adjencency_matrix.matmul(self.entity_emb).matmul(self.weight_0)

        last_layer = l0[:, entity].squeeze(1)
        user_entity_representation = self.gnn_last_layer_activation(last_layer)

        # Pred
        p = torch.einsum('bi,bi->bi', u_emb, user_entity_representation)

        return p





