import numpy as np


class JointUserMF:
    def __init__(self, n_users, n_movies, n_entities, k, lr=0.001, reg=0.0015):
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_entities = n_entities

        self.k = k
        self.lr = lr
        self.reg = reg

        self.U = np.random.rand(self.n_users, self.k)
        self.M = np.random.rand(self.n_movies, self.k)
        self.E = np.random.rand(self.n_entities, self.k)

    def predict(self, user, item, is_movie):
        return self.U[user] @ (self.M[item] if is_movie else self.E[item])

    def step(self, u, m, r, is_movie):
        error = self.predict(u, m, is_movie) - r

        Em = self.M if is_movie else self.E
        for k in range(self.k):
            u_gradient = error * Em[m][k]
            self.U[u][k] -= self.lr * (u_gradient - self.reg * Em[m][k])

            m_gradient = error * self.U[u][k]
            Em[m][k] -= self.lr * (m_gradient - self.reg * self.U[u][k])
