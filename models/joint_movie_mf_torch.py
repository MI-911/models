import torch as tt
import torch.nn as nn


class JointMovieMF(nn.Module):
    def __init__(self, n_users, n_movies, n_entities, k):
        super(JointMovieMF, self).__init__()

        self.n_users = n_users
        self.n_movies = n_movies
        self.n_entities = n_entities

        self.U = nn.Embedding(n_users, k)
        self.M = nn.Embedding(n_movies, k)
        self.E = nn.Embedding(n_entities, k)

    def forward(self, o, m, is_user):
        movie_embeddings = self.M(m)
        other_embeddings = self.choose_embeddings(o, is_user)

        predictions = (movie_embeddings * other_embeddings).sum(dim=1, keepdim=True)
        # predictions += self.Ub(users) + self.choose_biases(items, movie_map)

        return predictions.squeeze()

    def choose_embeddings(self, i, is_user):
        return (self.U(i % self.n_users) * is_user) + (self.E(i % self.n_entities) * (1 - is_user))