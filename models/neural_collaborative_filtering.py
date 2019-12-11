import torch
from torch import nn


class NCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_size):
        super(NCF, self).__init__()

        # Embeddings
        self.user_emb = nn.Linear(n_users, embedding_size, bias=False)
        self.item_emb = nn.Linear(n_items, embedding_size, bias=False)

        # GMF
        self.gmf_linear = nn.Linear(embedding_size, embedding_size, bias=False)
        self.gmf_activation = nn.Sigmoid()

        # MLP
        self.mlp_activation = nn.ReLU()
        self.mlp_layer_1 = nn.Linear(2 * embedding_size, embedding_size, bias=True)
        self.mlp_layer_2 = nn.Linear(embedding_size, embedding_size // 2, bias=True)

        # Fusion
        self.f_linear = nn.Linear(embedding_size + embedding_size // 2, 1, bias=False)
        self.f_activation = nn.Sigmoid()

    def forward(self, user_index, item_index):
        ue = self.user_emb(user_index)
        ie = self.item_emb(item_index)

        # GMF step
        gmf = self.gmf_activation(self.gmf_linear(ue * ie))

        # MLP step
        mlp = torch.cat((ue, ie), dim=1)
        mlp = self.mlp_activation(self.mlp_layer_1(mlp))
        mlp = self.mlp_activation(self.mlp_layer_2(mlp))

        # Fusion
        f = torch.cat((gmf, mlp), dim=1)
        f = self.f_activation(self.f_linear(f))

        return torch.flatten(f)

