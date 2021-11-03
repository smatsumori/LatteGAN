import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SentenceEncoder(nn.Module):
    def __init__(self, embedding_dim=1024):
        """
        Parameters
        ----------
        embedding_dim : int, optional
            Dimension of output sentence embedding "d_t", by default 1024
        """
        super().__init__()
        self.gru = nn.GRU(300,
                          embedding_dim // 2,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, words, lengths):
        """Return sentence embedding (B, D) from sequence of "word embeddings" (B, L, 300).

        Parameters
        ----------
        words : torch.tensor
            shape (B, seq_len, word_embedding_dim=300).
        lengths : torch.tensor
            shape (B,).
            List of each sequence lengths before padding.

        Returns
        -------
        h : torch.tensor
            shape (B, embedding_dim).
            Layer normalized state from last state of bidirectional GRU.
            In GeNeVa paper, it's represented with "d_t".
        """
        lengths = lengths.long()
        reorder = False
        sorted_len, indices = torch.sort(lengths, descending=True)
        if not torch.equal(sorted_len, lengths):
            _, reverse_sorting = torch.sort(indices)
            reorder = True
            words = words[indices]
            lengths = lengths[indices]

        lengths[lengths == 0] = 1

        packed_padded_sequence = pack_padded_sequence(words,
                                                      lengths,
                                                      batch_first=True)

        # Resets parameter data pointer so that they can use faster code paths.
        # Right now, this works only if the module is on the GPU and cuDNN is enabled.
        # Otherwise, it’s a no-op.
        self.gru.flatten_parameters()

        # h: shape (num_layers * num_directions, batch_size, embedding_dim)
        _, h = self.gru(packed_padded_sequence)
        h = h.permute(1, 0, 2).contiguous()
        h = h.view(-1, self.embedding_dim)

        if reorder:
            h = h[reverse_sorting]

        h = self.layer_norm(h)

        return h
