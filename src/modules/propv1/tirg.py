import torch
import torch.nn as nn

from .layers import conv3x3
from .normalization import BatchNorm2d, InstanceNorm2d
from .multihead_attention import MultiHeadAttention
from .spade import SPADE
from .relational_network import RelationalNetworkLite


class SourceTargetAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        key_dim,
        value_dim,
        d_model=512,
        nhead=1,
        bias=False,
        use_spectral_norm=False,
    ):
        super().__init__()

        # attntion module
        self.d_model = d_model
        self.attn = MultiHeadAttention(
            query_dim,
            key_dim,
            value_dim,
            d_model=d_model,
            nhead=nhead,
            bias=bias,
            use_spectral_norm=use_spectral_norm,
        )

    def _get_key_padding_mask_from_lengths(self, memory, lengths):
        batch_size, max_seq_len, _ = memory.size()

        key_padding_mask = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.bool,
            device=memory.device,
        )
        for i, idx in enumerate(lengths):
            key_padding_mask[i, idx:] = True

        return key_padding_mask

    def forward(self, query, memory, lengths):
        b, c, h, w = query.size()
        query = query.view(b, c, h * w)
        query = query.permute(0, 2, 1).contiguous()  # (B, HW, C)

        key_padding_mask = \
            self._get_key_padding_mask_from_lengths(memory, lengths)

        output = self.attn(
            query,
            memory,
            memory,
            key_padding_mask=key_padding_mask,
        )

        # output.shape = (B, HW, D)
        output = output.permute(0, 2, 1).contiguous()
        output = output.view(b, self.d_model, h, w)  # (B, C, H, W)

        return output


# TODO: fix behavior when use_relnet == True
class TIRG(nn.Module):
    def __init__(
        self,
        image_dim,
        text_dim,
        hidden_dim,
        norm="bn",
        sta="none",
        nhead=1,
        memory_dim=768,
        res_mask_post=False,
        multi_channel_gate=False,
        use_spectral_norm=False,
        use_relnet=False,
    ):
        super().__init__()

        # weighted coefficients
        self.w_gate = nn.Parameter(torch.ones(1))
        self.w_res = nn.Parameter(torch.zeros(1))

        in_channels = image_dim + text_dim

        if norm == "bn":
            Norm2d = BatchNorm2d
        elif norm == "in":
            Norm2d = InstanceNorm2d
        else:
            raise ValueError

        self.use_relnet = use_relnet
        if use_relnet:
            self.relnet = RelationalNetworkLite(
                image_dim=image_dim,
                text_dim=text_dim,
                out_dim=hidden_dim,
                maxlen=16,
                use_spectral_norm=use_spectral_norm,
                norm=norm,
            )
            in_channels = hidden_dim

        # concat source-target-attention text vectors
        self.sta_type = sta
        if self.sta_type in ["concat", "replace"]:
            self.sta_in = nn.Sequential(
                Norm2d(in_channels),
                nn.ReLU(),
            )
            self.sta = SourceTargetAttention(
                query_dim=in_channels,
                key_dim=memory_dim,
                value_dim=memory_dim,
                d_model=hidden_dim,
                nhead=nhead,
                bias=False,
                use_spectral_norm=use_spectral_norm,
            )
            if self.sta_type == "concat":
                in_channels += hidden_dim
            elif self.sta_type == "replace":
                in_channels = image_dim + hidden_dim
        elif self.sta_type == "none":
            if use_relnet:
                raise NotImplementedError
        else:
            raise ValueError

        # gating
        gate_dim = image_dim if multi_channel_gate else 1
        self.conv_gate = nn.Sequential(
            Norm2d(in_channels),
            nn.ReLU(),
            conv3x3(in_channels, hidden_dim, use_spectral_norm),
            Norm2d(hidden_dim),
            nn.ReLU(),
            conv3x3(hidden_dim, gate_dim, use_spectral_norm),
            nn.Sigmoid(),
        )

        # residual
        self.res_mask_post = res_mask_post
        self.conv_res = nn.Sequential(
            Norm2d(in_channels),
            nn.ReLU(),
            conv3x3(in_channels, hidden_dim, use_spectral_norm),
            Norm2d(hidden_dim),
            nn.ReLU(),
            conv3x3(hidden_dim, image_dim, use_spectral_norm),
        )

    def forward(self, x_img, x_txt, memories=None, lengths=None):
        if self.use_relnet:
            x = self.relnet(x_img, x_txt)
        else:
            x = torch.cat([x_img, x_txt], dim=1)

        # source target concat
        if self.sta_type in ["concat", "replace"]:
            query = self.sta_in(x)
            y = self.sta(query, memories, lengths)
            if self.sta_type == "concat":
                x = torch.cat([x, y], dim=1)
            elif self.sta_type == "replace":
                x = torch.cat([x_img, y], dim=1)
        elif self.sta_type == "none":
            if self.use_relnet:
                raise NotImplementedError
        else:
            raise ValueError

        # gating
        gate = self.conv_gate(x)

        # residual
        res = self.conv_res(x)
        if self.res_mask_post:
            res = (1 - gate) * res

        y = self.w_gate * (gate * x_img) + self.w_res * res
        return y, gate


class TIRGSPADE(nn.Module):
    def __init__(
        self,
        image_dim,
        text_dim,
        hidden_dim=512,
        norm="bn",
        sta="none",
        nhead=1,
        memory_dim=768,
        res_mask_post=False,
        multi_channel_gate=False,
        use_spectral_norm=False,
        use_relnet=False,
    ):
        super().__init__()

        # weighted coefficients
        self.w_gate = nn.Parameter(torch.ones(1))
        self.w_res = nn.Parameter(torch.zeros(1))

        in_channels = image_dim + text_dim

        if norm == "bn":
            Norm2d = BatchNorm2d
        elif norm == "in":
            Norm2d = InstanceNorm2d
        else:
            raise ValueError

        self.use_relnet = use_relnet
        if use_relnet:
            self.relnet = RelationalNetworkLite(
                image_dim=image_dim,
                text_dim=text_dim,
                out_dim=hidden_dim,
                maxlen=16,
                use_spectral_norm=use_spectral_norm,
                norm=norm,
            )
            in_channels = hidden_dim

        # concat source-target-attention text vectors
        self.sta_type = sta
        if self.sta_type in ["concat", "replace"]:
            self.sta_in = nn.Sequential(
                Norm2d(in_channels),
                nn.ReLU(),
            )
            self.sta = SourceTargetAttention(
                query_dim=in_channels,
                key_dim=memory_dim,
                value_dim=memory_dim,
                d_model=hidden_dim,
                nhead=nhead,
                bias=False,
                use_spectral_norm=use_spectral_norm,
            )
            if self.sta_type == "concat":
                # input of TIRG: [h, y, g]
                in_channels = image_dim + text_dim + hidden_dim
            elif self.sta_type == "replace":
                in_channels = image_dim + hidden_dim
        elif self.sta_type == "none":
            if use_relnet:
                raise NotImplementedError
        else:
            raise ValueError

        # gating
        gate_dim = image_dim if multi_channel_gate else 1
        self.conv_gate1 = nn.Sequential(
            Norm2d(in_channels),
            nn.ReLU(),
            conv3x3(in_channels, hidden_dim, use_spectral_norm),
        )
        self.conv_gate2 = nn.Sequential(
            Norm2d(hidden_dim),
            nn.ReLU(),
            conv3x3(hidden_dim, gate_dim, use_spectral_norm),
            nn.Sigmoid(),
        )

        # residual
        self.res_mask_post = res_mask_post
        self.spade_res1 = SPADE(
            hidden_dim,
            image_dim,
            kernel_size=3,
            padding=1,
            norm=norm,
            use_specral_norm=use_spectral_norm,
        )
        self.conv_res1 = nn.Sequential(
            nn.ReLU(),
            conv3x3(image_dim, image_dim, use_spectral_norm),
        )
        self.spade_res2 = SPADE(
            hidden_dim,
            image_dim,
            kernel_size=3,
            padding=1,
            norm=norm,
            use_specral_norm=use_spectral_norm,
        )
        self.conv_res2 = nn.Sequential(
            nn.ReLU(),
            conv3x3(image_dim, image_dim, use_spectral_norm),
        )

    def forward(self, x_img, x_txt, memories=None, lengths=None):
        if self.use_relnet:
            x = self.relnet(x_img, x_txt)
        else:
            x = torch.cat([x_img, x_txt], dim=1)

        # source target concat
        if self.sta_type in ["concat", "replace"]:
            query = self.sta_in(x)
            y = self.sta(query, memories, lengths)
            if self.sta_type == "concat":
                x = torch.cat([x_img, x_txt, y], dim=1)
            elif self.sta_type == "replace":
                x = torch.cat([x_img, y], dim=1)
        elif self.sta_type == "none":
            if self.use_relnet:
                raise NotImplementedError
        else:
            raise ValueError

        # gating
        pre_gate = self.conv_gate1(x)
        gate = self.conv_gate2(pre_gate)

        # residual
        res = self.spade_res1(x_img, pre_gate)
        res = self.conv_res1(res)
        res = self.spade_res2(res, pre_gate)
        res = self.conv_res2(res)
        if self.res_mask_post:
            res = (1 - gate) * res

        y = self.w_gate * (gate * x_img) + self.w_res * res
        return y, gate
