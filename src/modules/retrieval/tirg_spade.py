import torch
import torch.nn as nn

from .tirg import SourceTargetAttention
from .spade import SPADE


class TIRGSPADE(nn.Module):
    def __init__(
        self,
        image_dim,
        text_dim,
        hidden_dim,
        sta_concat=False,
        use_pos_emb=False,
        res_mask_post=False,
        multi_channel_gate=False,
        num_objects=58,
        **kwargs,
    ):
        super().__init__()

        self.w_gate = nn.Parameter(torch.ones(1))
        self.w_res = nn.Parameter(torch.zeros(1))

        in_channels = image_dim + text_dim

        self.sta_concat = sta_concat
        if sta_concat:
            self.st_attn = SourceTargetAttention(
                in_channels,
                text_dim,
                hidden_dim,
                use_pos_emb,
            )
            in_channels += hidden_dim

        # gating 1/2
        self.conv_gate1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )

        # gating 2/2 (single or multi channels)
        self.multi_channel_gate = multi_channel_gate
        gate_dim = image_dim if multi_channel_gate else 1
        self.conv_gate2 = nn.Sequential(
            nn.Conv2d(hidden_dim, gate_dim, 3, padding=1),
            nn.Sigmoid(),
        )

        # detection added objects from gating pass
        # --> branch gating 1/2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.detector = nn.Linear(hidden_dim, num_objects)

        # residual
        self.res_mask_post = res_mask_post

        self.conv_res1 = nn.Conv2d(image_dim, image_dim, 3, padding=1)
        self.spade_res1 = SPADE(hidden_dim, image_dim, 3, padding=1)
        self.relu_res1 = nn.ReLU()
        self.conv_res2 = nn.Conv2d(image_dim, image_dim, 3, padding=1)
        self.spade_res2 = SPADE(hidden_dim, image_dim, 3, padding=1)
        self.relu_res2 = nn.ReLU()

    def forward(
        self,
        x_img,
        x_txt,
        memory=None,
        lengths=None,
        key_padding_mask=None,
    ):
        b, _, h, w = x_img.size()
        x_txt = x_txt.view(b, -1, 1, 1).repeat(1, 1, h, w)
        x = torch.cat([x_img, x_txt], dim=1)

        # source target concat
        if self.sta_concat:
            y_txt = self.st_attn(x, memory, lengths, key_padding_mask)
            x = torch.cat([x, y_txt], dim=1)

        # gating
        gate1 = self.conv_gate1(x)
        gate2 = self.conv_gate2(gate1)

        # added object detection
        detect_from_gate = self.pool(gate1).squeeze(dim=3).squeeze(dim=2)
        detect_from_gate = self.detector(detect_from_gate)

        # residual
        res = self.conv_res1(x_img)
        res = self.spade_res1(res, gate1)
        res = self.relu_res1(res)
        res = self.conv_res2(res)
        res = self.spade_res2(res, gate1)
        res = self.relu_res2(res)
        if self.res_mask_post:
            res = (1 - gate2) * res

        y = self.w_gate * (gate2 * x_img) + self.w_res * res
        return y, gate2, detect_from_gate
