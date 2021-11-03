import torch.nn as nn


class DialogEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=False,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, dt, hidden):
        self.rnn.flatten_parameters()

        ht, hidden = self.rnn(dt, hidden)
        ht = self.layer_norm(ht)

        return ht, hidden
