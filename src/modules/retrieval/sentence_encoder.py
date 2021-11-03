import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import BertModel


class BERTSentenceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.train()

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        y = outputs.last_hidden_state[:, 0]

        key_padding_mask = token_type_ids.detach().clone()
        # [CLS, SA1, SA2, ..., SEP, SB1, SB2, ..., SEP, PAD, PAD, ...]
        # [  1,   0,   0, ...,   0,   1,   1, ...,   1,   0,   0, ...]
        key_padding_mask[:, 0] = 1
        key_padding_mask = torch.logical_not(key_padding_mask)

        return outputs.last_hidden_state, y, key_padding_mask


class SentenceEncoder(nn.Module):
    def __init__(self, embedding_dim=1024):
        super().__init__()
        self.rnn = nn.GRU(
            300,
            embedding_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim // 2)
        self.embedding_dim = embedding_dim

    def forward(self, embs, lengths, h_0=None):
        # sort data for packing padded sequence
        lengths = lengths.long()
        reorder = False
        sorted_len, indices = torch.sort(lengths, descending=True)
        if not torch.equal(sorted_len, lengths):
            _, reverse_sorting = torch.sort(indices)
            reorder = True
            embs = embs[indices]
            lengths = lengths[indices]
            if h_0 is not None:
                h_0 = h_0[:, indices]

        # null text inputs will be 1 <pad> token
        lengths[lengths == 0] = 1

        packed_padded_sequence = pack_padded_sequence(
            embs, lengths, batch_first=True
        )

        # Resets parameter data pointer so that they can use faster code paths.
        # Right now, this works only if the module is on the GPU and cuDNN is enabled.
        # Otherwise, it’s a no-op.
        self.rnn.flatten_parameters()

        # output: shape (batch_size, seq_len, num_directions * hidden_dim)
        # h_n: shape (num_layers * num_directions, batch_size, hidden_dim)
        output, h_n = self.rnn(packed_padded_sequence, h_0)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.norm1(output)
        h_n = self.norm2(h_n)

        # extract hidden state as context
        h = h_n.permute(1, 0, 2).contiguous()
        h = h.view(-1, self.embedding_dim)

        # reorder output data if needed
        if reorder:
            output = output[reverse_sorting]
            h = h[reverse_sorting]
            h_n = h_n[:, reverse_sorting]

        return output, h, h_n
