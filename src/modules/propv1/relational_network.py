import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from .layers import conv1x1, conv3x3
from .normalization import BatchNorm2d, InstanceNorm2d


class RelationalNetwork(nn.Module):
    def __init__(
        self,
        image_dim,
        text_dim,
        out_dim,
        maxlen=256,
        use_spectral_norm=False,
        norm="in",
    ):
        super().__init__()

        if norm == "bn":
            Norm2d = BatchNorm2d
        elif norm == "in":
            Norm2d = InstanceNorm2d
        else:
            raise ValueError

        self.pos_emb = nn.Embedding(maxlen, image_dim)
        self.norm_image = nn.Sequential(
            Norm2d(image_dim),
            nn.ReLU(),
        )
        self.norm_text = nn.Sequential(
            Norm2d(text_dim),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            conv1x1(image_dim * 2 + text_dim, out_dim, use_spectral_norm),
            Norm2d(out_dim),
            nn.ReLU(),
            conv1x1(out_dim, out_dim, use_spectral_norm),
        )

        if use_spectral_norm:
            nn.init.orthogonal_(self.pos_emb.weight.data)
            self.pos_emb = spectral_norm(self.pos_emb)
        else:
            nn.init.normal_(self.pos_emb.weight.data, 0.0, 0.02)

    def add_positional_embedding(self, x):
        # x.shape = (B, C, H, W)
        b, c, h, w = x.size()

        tokens = torch.arange(h * w, dtype=torch.long, device=x.device)
        tokens = tokens.unsqueeze(dim=0)  # --> (1, HW)
        tokens = tokens.repeat(b, 1)  # --> (B, HW)
        pos = self.pos_emb(tokens)  # --> (B, HW, C)
        pos = pos.permute(0, 2, 1).contiguous()  # --> (B, C, HW)
        pos = pos.view(b, c, h, w)

        return x + pos

    def forward(self, x, y):
        # positional embedding
        x = self.add_positional_embedding(x)

        # normalization
        x = self.norm_image(x)
        y = self.norm_text(y)

        # x.shape = (B, C, H, W)
        b, c, h, w = x.size()

        x = x.view(b, c, h * w)  # --> (B, C, HW)

        # prepare pairs of (xi, xj)
        x_i = x.unsqueeze(dim=2)  # --> (B, C, 1, HW)
        x_i = x_i.repeat(1, 1, h * w, 1)  # --> (B, C, (HW), HW)
        x_j = x.unsqueeze(dim=3)  # --> (B, C, HW, 1)
        x_j = x_j.repeat(1, 1, 1, h * w)  # --> (B, C, HW, (HW))

        # y.shape = (B, D, H, W)
        _, d, _, _ = y.size()
        y = y.view(b, d, h * w)
        y = y.unsqueeze(dim=3)
        y = y.repeat(1, 1, 1, h * w)  # --> (B, D, HW, (HW))

        # concat and transform
        x = torch.cat([x_i, x_j, y], dim=1)  # --> (B, C+C+D, HW, HW)
        x = self.conv(x)

        # sum pooling along each positions, and reshape to original shape
        x = torch.sum(x, dim=3)
        x = x.view(b, -1, h, w)

        return x


class RelationalNetworkLite(nn.Module):
    def __init__(
        self,
        image_dim,
        text_dim,
        out_dim,
        maxlen=16,
        use_spectral_norm=False,
        norm="in",
    ):
        super().__init__()
        assert out_dim % 2 == 0

        if norm == "bn":
            Norm2d = BatchNorm2d
        elif norm == "in":
            Norm2d = InstanceNorm2d
        else:
            raise ValueError

        self.h_rel_net = RelationalNetwork(
            image_dim,
            text_dim,
            out_dim // 2,
            maxlen,
            use_spectral_norm,
            norm,
        )
        self.w_rel_net = RelationalNetwork(
            image_dim,
            text_dim,
            out_dim // 2,
            maxlen,
            use_spectral_norm,
            norm,
        )
        self.last_conv = nn.Sequential(
            Norm2d(out_dim),
            nn.ReLU(),
            conv3x3(out_dim, out_dim, use_spectral_norm),
        )

    def forward(self, x, y):
        # x.shape = (B, C, H, W)
        b, c, h, w = x.size()
        xh = x.mean(dim=3, keepdim=True)  # --> (B, C, H, 1)
        xw = x.mean(dim=2, keepdim=True)  # --> (B, C, 1, W)

        # y.shape = (B, D, H, W)
        _, d, _, _ = y.size()
        yh = y.mean(dim=3, keepdim=True)  # --> (B, D, H, 1)
        yw = y.mean(dim=2, keepdim=True)  # --> (B, D, 1, W)

        # relational encode along height or width
        xh = self.h_rel_net(xh, yh)  # --> (B, O, H, 1)
        xw = self.w_rel_net(xw, yw)  # --> (B, O, 1, W)

        # repeat and concat
        xh = xh.repeat(1, 1, 1, w)
        xw = xw.repeat(1, 1, h, 1)
        x = torch.cat([xh, xw], dim=1)
        x = self.last_conv(x)

        return x
