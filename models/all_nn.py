import torch as tt
import torch.nn as nn
import torch.nn.functional as ff


class NeuralNetworkAll(nn.Module):
    def __init__(self, n_movies, n_entities, n_fc1, n_fc2):
        super(NeuralNetworkAll, self).__init__()
        self.input = nn.Linear(n_movies + n_entities, n_fc1)
        self.fc1 = nn.Linear(n_fc1, n_fc2)
        self.fc2 = nn.Linear(n_fc2, n_movies)

    def forward(self, one_hot):
        x = ff.sigmoid(self.input(one_hot))
        x = ff.sigmoid(self.fc1(x))
        out = ff.sigmoid(self.fc2(x))

        return out