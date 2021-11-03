import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .layers import ResBlockDown, conv1x1
from .multihead_attention import MultiHeadAttention


def linear(in_dim, out_dim, use_spectral_norm=False):
    n = nn.Linear(in_dim, out_dim, bias=False)
    if use_spectral_norm:
        nn.init.orthogonal_(n.weight.data)
        n = spectral_norm(n)
    else:
        nn.init.normal_(n.weight.data, 0.0, 0.02)
    return n


class SourceTargetAttentionProjection(nn.Module):
    """source target attention applied to projection module.

    x: image (difference) feature, shape=(B, 1024, 4, 4).
    memories: word embeddings of instruction at timestep t, shape=(B, l, 768).

    query, key, and value are calulated as following:
    q = Wq * memories (q.shape=(B, l, 512))
    k, v = Wk * x, Wv * x (k,v.shape=(B, 4^2, 512))

    calulate word embedding guided image feature using MultiheadAttention.
    (e.g. "cat" embedding guides cat image feature in feature map if cat image are generated.)
    out = Wo * (softmax(q * k) * v) (out.shape=(B, l, 512))

    If image feature are successfully gathered, inner-product with query will be high, it means more realistic (higher adv).
    prod = q * out (prod.shape=(B, l))
    prod will be pooled along length except padding position.
    """

    def __init__(
        self,
        query_dim,  # word embeddings
        key_dim,  # visual difference
        value_dim,  # visual difference
        d_model=512,
        nhead=8,
        bias=False,
        use_spectral_norm=True,
        use_gate_for_stap=False,
    ):
        super().__init__()
        # attntion module
        self.embed_query = linear(query_dim, d_model, use_spectral_norm)
        self.attn = MultiHeadAttention(
            d_model,  # query is embedded into d_model dimension
            key_dim,
            value_dim,
            d_model=d_model,
            nhead=nhead,
            bias=bias,
            use_spectral_norm=use_spectral_norm,
        )
        # projection (query * value)
        self.fc_value = linear(d_model, d_model, use_spectral_norm)

        # gating projection results by sigmoid(f([y_cls, ei]))
        self.use_gate_for_stap = use_gate_for_stap
        if use_gate_for_stap:
            self.gating = nn.Sequential(
                linear(query_dim + query_dim, 1, use_spectral_norm),
                nn.Sigmoid(),
            )

    def _get_padding_mask_from_lengths(self, memory, lengths):
        batch_size, max_seq_len, _ = memory.size()
        padding_mask = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.bool,
            device=memory.device,
        )
        for i, idx in enumerate(lengths):
            padding_mask[i, idx:] = True
        return padding_mask

    def forward(self, x, memory, lengths, y=None):
        # embed query
        query = self.embed_query(memory)  # (B, L, d_model)

        # reshape source of key and value
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        x = x.permute(0, 2, 1).contiguous()  # (B, HW, C)

        # STA, value.shape = (B, L, d_model)
        value = self.attn(query, x, x, key_padding_mask=None)
        value = F.relu(self.fc_value(value))

        # projection
        prod = torch.sum(query * value, dim=2)  # (B, L)

        # gating if needed
        if self.use_gate_for_stap:
            # y.shape = (B, D) --> (B, L, D)
            y = y.unsqueeze(dim=1).repeat(1, memory.size(1), 1)
            y = torch.cat([y, memory], dim=2)
            y = self.gating(y).squeeze(dim=2)  # (B, L)
            prod = y * prod

        padding_mask = self._get_padding_mask_from_lengths(memory, lengths)
        prod = prod.masked_fill(padding_mask, 0.0)
        prod = torch.sum(prod, dim=1, keepdim=True)

        return prod


class CoAttentionProjection(nn.Module):
    def __init__(
        self,
        image_dim,
        text_dim,
        d_model=512,
        nhead=8,
        bias=False,
        use_spectral_norm=True,
        use_gate=False,
    ):
        super().__init__()

        # embed into d_model
        self.fc_image = linear(image_dim, d_model, use_spectral_norm)
        self.fc_text = linear(text_dim, d_model, use_spectral_norm)

        # attntion module
        # tii: (Q=text, K=image, V=image) --> for AR
        # itt: (Q=image, K=text, V=text) --> for AP
        self.attn_tii = MultiHeadAttention(
            d_model,
            d_model,
            d_model,
            d_model=d_model,
            nhead=nhead,
            bias=bias,
            use_spectral_norm=use_spectral_norm,
        )
        self.attn_itt = MultiHeadAttention(
            d_model,
            d_model,
            d_model,
            d_model=d_model,
            nhead=nhead,
            bias=bias,
            use_spectral_norm=use_spectral_norm,
        )

        # projection (query * value)
        self.fc_value_tii = nn.Sequential(
            linear(d_model, d_model, use_spectral_norm),
            nn.ReLU(),
        )
        self.fc_value_itt = nn.Sequential(
            linear(d_model, d_model, use_spectral_norm),
            nn.ReLU(),
        )

        # gating projection results
        self.use_gate = use_gate
        if use_gate:
            self.gating_image = nn.Sequential(
                linear(image_dim + image_dim, 1, use_spectral_norm),
                nn.Sigmoid(),
            )
            self.gating_text = nn.Sequential(
                linear(text_dim + text_dim, 1, use_spectral_norm),
                nn.Sigmoid(),
            )

    def _get_padding_mask_from_lengths(self, memory, lengths):
        batch_size, max_seq_len, _ = memory.size()
        padding_mask = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.bool,
            device=memory.device,
        )
        for i, idx in enumerate(lengths):
            padding_mask[i, idx:] = True
        return padding_mask

    def forward(
        self,
        image,
        memory,
        lengths,
        y=None,
    ):
        # reshape image
        b, c, h, w = image.size()
        image = image.view(b, c, h * w)
        image = image.permute(0, 2, 1).contiguous()  # (B, HW, C)

        # append global image vector: h
        h = torch.sum(image, dim=1, keepdim=True)  # (B, 1, C)
        image = torch.cat([h, image], dim=1)  # (B, 1+HW, C)

        # embed attn inputs
        embed_image = self.fc_image(image)
        embed_memory = self.fc_text(memory)

        # padding mask
        padding_mask = self._get_padding_mask_from_lengths(memory, lengths)

        # CoAttention
        # value_tii.shape = (B, L, d_model)
        # value_itt.shape = (B, HW, d_model)
        value_tii = self.fc_value_tii(
            self.attn_tii(
                embed_memory,
                embed_image,
                embed_image,
                key_padding_mask=None,
            )
        )
        value_itt = self.fc_value_itt(
            self.attn_itt(
                embed_image,
                embed_memory,
                embed_memory,
                key_padding_mask=padding_mask,
            )
        )

        # projection
        prod_tii = torch.sum(embed_memory * value_tii, dim=2)  # (B, L)
        prod_itt = torch.sum(embed_image * value_itt, dim=2)  # (B, 1+HW)

        # gating if needed
        if self.use_gate:
            # prod_tii gate
            # y.shape = (B, D) --> (B, L, D)
            y = y.unsqueeze(dim=1).repeat(1, memory.size(1), 1)
            y = torch.cat([y, memory], dim=2)
            y = self.gating_text(y).squeeze(dim=2)  # (B, L)
            prod_tii = y * prod_tii

            # prod_itt gate
            h = h.repeat(1, image.size(1), 1)  # (B, 1+HW, C)
            h = torch.cat([h, image], dim=2)
            h = self.gating_image(h).squeeze(dim=2)  # (B, 1+HW)
            prod_itt = h * prod_itt

        # mask padded projection
        prod_tii = prod_tii.masked_fill(padding_mask, 0.0)
        prod = torch.sum(prod_tii, dim=1, keepdim=True) + \
            torch.sum(prod_itt, dim=1, keepdim=True)

        return prod


class STAPDiscriminator(nn.Module):
    def __init__(
        self,
        condition_dim=768,
        discriminator_sn=True,
        aux_detection_dim=58,
        fusion="subtract",
        d_model=512,
        nhead=8,
        use_gate_for_stap=False,
        use_co_attention=False,
    ):
        super().__init__()

        # encoder for both source and target image
        self.resdown1 = ResBlockDown(
            3, 64, use_spectral_norm=discriminator_sn, is_first_block=True)
        self.resdown2 = ResBlockDown(
            64, 128, use_spectral_norm=discriminator_sn)
        self.resdown3 = ResBlockDown(
            128, 256, use_spectral_norm=discriminator_sn)

        # discriminator from feature difference
        self.fusion = fusion
        if fusion == "subtract":
            # [xt - xs]
            in_channels = 256
        elif fusion == "concat":
            # [xs, xt]
            in_channels = 256 * 2
        elif fusion == "all":
            # [xs, xt, xt - xs]
            # inspired by sentence relation classification
            in_channels = 256 * 3
        else:
            raise ValueError

        self.resdown4 = ResBlockDown(
            in_channels, 512, use_spectral_norm=discriminator_sn)
        self.resdown5 = ResBlockDown(
            512, 1024, use_spectral_norm=discriminator_sn)
        self.resdown6 = ResBlockDown(
            1024, 1024, use_spectral_norm=discriminator_sn, use_downsample=False)

        # adversarial
        self.linear = conv1x1(1024, 1, discriminator_sn)

        # projection
        self.use_co_attention = use_co_attention
        if use_co_attention:
            self.coattn_projector = CoAttentionProjection(
                image_dim=1024,
                text_dim=condition_dim,
                d_model=d_model,
                bias=False,
                use_spectral_norm=discriminator_sn,
                use_gate=use_gate_for_stap,
            )
        else:
            self.sta_projector = SourceTargetAttentionProjection(
                condition_dim,
                1024,
                1024,
                d_model=d_model,
                nhead=nhead,
                bias=False,
                use_spectral_norm=discriminator_sn,
                use_gate_for_stap=use_gate_for_stap,
            )

        # auxiliary object detector
        self.aux_detector = nn.Sequential(
            conv1x1(1024, 256, discriminator_sn),
            nn.ReLU(),
            conv1x1(256, aux_detection_dim, discriminator_sn),
        )

    def forward(
        self,
        x_src,
        x_tgt,
        memories,
        lengths,
        y=None,
    ):
        """forward paired data to discriminate real or fake.

        Args:
            x_src (torch.Tensor): shape=(B, 3, 128, 128).
            x_tgt (torch.Tensor): real/fake image of timestep t, shape=(B, 3, 128, 128).
            memories (torch.Tensor): word embeddings of instruction at timestep t, shape=(B, l, D).
            lengths (torch.Tensor): text length of each samples in minibatch, l_i <= l. shape=(B, ).
            y (torch.Tensor): text embedding of instruction at timestep t, shape=(B, D).

        Returns:
            adv (torch.Tensor): adversarial logits, shape=(B, 1).
            aux (torch.Tensor): auxliary detection logits of added object at timestep t, shape=(B, num_objects).
        """
        # encoder for both source and target image
        # source image
        x_src = self.resdown1(x_src)
        x_src = self.resdown2(x_src)
        x_src = self.resdown3(x_src)
        # target image
        x_tgt = self.resdown1(x_tgt)
        x_tgt = self.resdown2(x_tgt)
        x_tgt = self.resdown3(x_tgt)

        # discriminator from feature difference
        if self.fusion == "subtract":
            x = x_tgt - x_src
        elif self.fusion == "concat":
            x = torch.cat([x_src, x_tgt], dim=1)
        elif self.fusion == "all":
            dx = x_tgt - x_src
            x = torch.cat([x_src, x_tgt, dx], dim=1)

        x = self.resdown4(x)
        x = self.resdown5(x)
        x = self.resdown6(x)
        x = F.relu(x)
        # x.shape = (B, 1024, 4, 4)
        # h.shape = (B, 1024, 1, 1)
        h = torch.sum(x, dim=(2, 3), keepdim=True)

        # adversarial
        # adv.shape = (B, 1)
        adv = self.linear(h)
        adv = adv.squeeze(3).squeeze(2)

        # source target attention projection
        # word embeddings as query: memories with lengths
        # image difference as key, value: x, x.shape=(B, 1024, 4, 4).
        # y.shape = (B, 1)
        if self.use_co_attention:
            y = self.coattn_projector(
                image=x,
                memory=memories,
                lengths=lengths,
                y=y,
            )
        else:
            y = self.sta_projector(x, memories, lengths, y)
        adv = adv + y

        # auxiliary object detector
        aux = self.aux_detector(h)
        aux = aux.squeeze(3).squeeze(2)

        return adv, aux
