import torch.nn as nn


class ConditionEncoder(nn.Module):
    """
    Original GeNeVA
        text_features (B, embedding_dim=1024)
        ConditionEncoder: text_features --> fused_features (B, projected_text_dim=input_dim=1024)
        RNN: h_(t-1), fused_features --> output (B, hidden_dim=1024)

    Single-turn GeNeVA
        text_features (B, embedding_dim=1024)
        ConditionEncoder: text_features --> fused_features (B, hidden_dim=1024)
    """

    def __init__(self, embedding_dim=1024, hidden_dim=1024):
        """
        Parameters
        ----------
        embedding_dim : int, optional
            Dimension of input text embedding "d_t", by default 1024
        hidden_dim : int, optional
            Dimension of output text embedding "h_t", by default 1024
        """
        super().__init__()
        self.text_projection = nn.Linear(embedding_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, text_features):
        """Given raw text_features from SentenceEncoder, return transformed it.

        Parameters
        ----------
        text_features : torch.tensor
            shape (B, C).
            C = embedding_dim (default 1024).
            Text embeded feature from Sentence Encoder.

        Returns
        -------
        fused_features : torch.tensor
            shape (B, C).
            C = hidden_dim (default 1024).
            In original GeNeVA, this output is feeded GRU across an episode, then become h_t.
            In this single-turn setting, this output is considered as h_t directly.
        """
        text_features = self.text_projection(text_features)
        fused_features = self.bn(text_features)
        return fused_features
