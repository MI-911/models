import torch as tt
import torch.nn as nn
import torch.nn.functional as ff


class JointUserMF(nn.Module):
    def __init__(self, n_users, n_movies, n_entities, k):
        super(JointUserMF, self).__init__()
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_entities = n_entities
        self.k = k

        # Embedding layers
        self.U = nn.Embedding(self.n_users, self.k)
        self.M = nn.Embedding(self.n_movies + self.n_entities, self.k)
        # self.E = nn.Embedding(self.n_entities, self.k)

        # Biases
        self.Ub = nn.Embedding(self.n_users, 1)
        self.Mb = nn.Embedding(self.n_movies + self.n_entities, 1)
        # self.Eb = nn.Embedding(self.n_entities, 1)

    def forward(self, users, items, movie_map):
        # The movie map determines, for every user-item pair, whether the
        # item is a movie or an entity (1 for movies, 0 for entities)
        # Choose the proper embeddings from this information.

        u_embeddings = self.U(users)
        i_embeddings = self.M(items)

        # Dot product all vectors pairwise, add bias and return
        predictions = (u_embeddings * i_embeddings).sum(dim=1, keepdim=True)
        predictions += self.Ub(users) + self.Mb(items)

        return predictions.squeeze()

    def choose_embeddings(self, i, is_movie):
        return (self.M(i % self.n_movies) * is_movie) + (self.E(i % self.n_entities) * (1 - is_movie))

    def choose_biases(self, i, is_movie):
        return (self.Mb(i % self.n_movies) * is_movie) + (self.Eb(i % self.n_entities) * (1 - is_movie))
