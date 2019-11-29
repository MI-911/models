import torch
from torch import nn


class MF(nn.Module):
    def __init__(self, user_count, item_count, latent_factors):
        super(MF, self).__init__()
        self.users = nn.Embedding(user_count, latent_factors, sparse=True)
        self.items = nn.Embedding(item_count, latent_factors, sparse=True)

    def forward(self, user_id, item_id):
        return (self.users(user_id) * self.items(item_id)).sum(1)


class MFExtended(nn.Module):
    def __init__(self, user_count, item_count, latent_factors):
        super(MFExtended, self).__init__()
        self.users = nn.Embedding(user_count, latent_factors, sparse=True)
        self.items = nn.Embedding(item_count, latent_factors, sparse=True)

        self.ub = nn.Embedding(user_count, 1, sparse=True)
        self.ib = nn.Embedding(item_count, 1, sparse=True)

        self.activation = nn.Sigmoid()

    def forward(self, user_id, item_id):
        pred = self.ub(user_id) + self.ib(item_id)
        pred += (self.users(user_id) * self.items(item_id)).sum(1, keepdim=True)
        return self.activation(pred.squeeze())
