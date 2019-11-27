import torch as tt
import torch.nn as nn
import torch.nn.functional as ff
import torch.optim as optimizers


class JointUserMF(nn.Module):
    def __init__(self, n_users, n_movies, n_entities, k):
        super(JointUserMF, self).__init__()
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_entities = n_entities
        self.k = k

        # Embedding layers
        self.U = nn.Embedding(self.n_users, self.k)
        self.M = nn.Embedding(self.n_movies, self.k)
        self.E = nn.Embedding(self.n_entities, self.k)

        # Biases
        self.Ub = nn.Embedding(self.n_users, 1)
        self.Mb = nn.Embedding(self.n_movies, 1)
        self.Eb = nn.Embedding(self.n_entities, 1)

    def forward(self, users, items):
        u_embeddings = self.U(users)
        i_embeddings = self.I(items)

        # Dot product all vectors pairwise, add bias and return
        predictions = (u_embeddings * i_embeddings).sum(dim=1, keepdim=True)
        predictions += self.Ub(users) + self.Ib(items)

        return predictions.squeeze()
