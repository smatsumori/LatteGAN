import torch
import torch.nn as nn

from .layers import conv3x3, conv1x1
from .normalization import BatchNorm2d, InstanceNorm2d


class TIDG(nn.Module):
    def __init__(
        self,
        image_dim,
        text_dim,
        hidden_dim,
        norm="bn",
        res_mask_post=False,
        multi_channel_gate=False,
        use_spectral_norm=False,
    ):
        super().__init__()

        # weighted coefficients
        self.w_gate = nn.Parameter(torch.ones(1))
        self.w_res = nn.Parameter(torch.zeros(1))

        # text embedding
        self.emb = conv1x1(text_dim, hidden_dim, use_spectral_norm)

        in_channels = image_dim + hidden_dim

        if norm == "bn":
            Norm2d = BatchNorm2d
        elif norm == "in":
            Norm2d = InstanceNorm2d
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

    def forward(self, xs, xt, y):
        y = y.unsqueeze(dim=2).unsqueeze(dim=3)
        y = self.emb(y)

        dxt = xt - xs
        _, _, h, w = dxt.size()
        y = y.repeat(1, 1, h, w)
        dxty = torch.cat([dxt, y], dim=1)

        # gating
        gate = self.conv_gate(dxty)

        # residual
        if self.res_mask_post:
            res = (1 - gate) * xt

        output = self.w_gate * (gate * xs) + self.w_res * res
        return output, gate
