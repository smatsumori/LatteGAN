import math

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class ScaledDotProductAttention(nn.Module):
    def forward(self, query, key, value, key_padding_mask=None):
        """scaled dot product attention.

        Args:
            query (torch.tensor): shape=(B, S, D)
            key (torch.tensor): shape=(B, L, D)
            value (torch.tensor): shape=(B, L, D)
            key_padding_mask (torch.tensor, optional): shape=(B, L) for padding. Defaults to None.

        Returns:
            torch.tensor: shape=(B, S, D)
        """
        dk = query.size(2)
        # scores.shape = (B, S, L)
        scores = query.matmul(key.transpose(2, 1)) / math.sqrt(dk)

        if key_padding_mask is not None:
            slen = query.size(1)
            key_padding_mask = key_padding_mask.unsqueeze(dim=1)
            # key_padding_mask.shape = (B, S, L)
            key_padding_mask = key_padding_mask.repeat(1, slen, 1)
            scores = scores.masked_fill(key_padding_mask, -1e9)

        attention = F.softmax(scores, dim=2)
        value = attention.matmul(value)
        return value, attention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        query_features,
        key_features,
        value_features,
        d_model=512,
        nhead=1,
        bias=False,
        use_spectral_norm=False,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError

        self.fc_q = nn.Linear(query_features, d_model, bias)
        self.fc_k = nn.Linear(key_features, d_model, bias)
        self.fc_v = nn.Linear(value_features, d_model, bias)

        if use_spectral_norm:
            self.fc_q = spectral_norm(self.fc_q)
            self.fc_k = spectral_norm(self.fc_k)
            self.fc_v = spectral_norm(self.fc_v)
            nn.init.orthogonal_(self.fc_q.weight.data)
            nn.init.orthogonal_(self.fc_k.weight.data)
            nn.init.orthogonal_(self.fc_v.weight.data)
        else:
            nn.init.normal_(self.fc_q.weight.data, 0.0, 0.02)
            nn.init.normal_(self.fc_k.weight.data, 0.0, 0.02)
            nn.init.normal_(self.fc_v.weight.data, 0.0, 0.02)

        if bias:
            nn.init.constant_(self.fc_q.bias.data, 0)
            nn.init.constant_(self.fc_k.bias.data, 0)
            nn.init.constant_(self.fc_v.bias.data, 0)

        self.nhead = nhead
        self.scaled_dot_product_attention = ScaledDotProductAttention()

    def forward(self, q, k, v, key_padding_mask=None):
        """multihead attention.

        Args:
            q (torch.tensor): shape=(B, S, D1)
            k (torch.tensor): shape=(B, L, D2)
            v (torch.tensor): shape=(B, L, D3)
            key_padding_mask (torch.tensor, optional): shape=(B, L) for padding. Defaults to None.

        Returns:
            torch.tersor: shape=(B, S, D)
        """
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.repeat(self.nhead, 1)

        y, _ = self.scaled_dot_product_attention(q, k, v, key_padding_mask)
        y = self._reshape_from_batches(y)

        return y

    def _reshape_to_batches(self, x):
        """reshape batch tensor to multihead batch tensor.

        Args:
            x (torch.tensor): shape=(B, L, D).

        Returns:
            torch.tensor: shape=(B * N, L, D // N)
        """
        batch_size, seq_len, d_model = x.size()
        sub_d_model = d_model // self.nhead
        x = x.reshape(batch_size, seq_len, self.nhead, sub_d_model)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size * self.nhead, seq_len, sub_d_model)
        return x

    def _reshape_from_batches(self, x):
        """reshape multihead batch tensor to batch tensor.

        Args:
            x (torch.tensor): shape=(B * N, L, D // N).

        Returns:
            torch.tensor: shape=(B, L, D).
        """
        nhead_batch_size, seq_len, sub_d_model = x.size()
        batch_size = nhead_batch_size // self.nhead
        x = x.view(batch_size, self.nhead, seq_len, sub_d_model)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, seq_len, sub_d_model * self.nhead)
        return x
