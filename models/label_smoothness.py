import torch
from torch import nn


class GNN(nn.Module):
    def __init__(self, n_users, n_relations, n_entities, adjacency_matrix, latent_dim=8, cuda=False):
        super(GNN, self).__init__()

        self.user_emb = nn.Embedding(n_users, latent_dim)
        self.relation_emb = nn.Parameter(torch.rand((latent_dim, n_relations)), requires_grad=True)
        self.entity_emb = nn.Parameter(torch.rand((n_entities, latent_dim)), requires_grad=True)
        self.adjaceny_matrix = adjacency_matrix

    def forward(self, user, entity):
        u_emb = self.user_emb(user)
        relation_weights = u_emb.dot(self.relation_emb)


        pass