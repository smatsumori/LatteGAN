import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .normalization import ConditionalBatchNorm2d, BatchNorm2d
from .normalization import AdaptiveInstanceNorm2d, InstanceNorm2d


def conv1x1(in_ch, out_ch, use_spectral_norm=False):
    n = nn.Conv2d(in_ch, out_ch, 1, bias=False)
    if use_spectral_norm:
        nn.init.orthogonal_(n.weight.data)
        n = spectral_norm(n)
    else:
        nn.init.normal_(n.weight.data, 0.0, 0.02)
    return n


def conv3x3(in_ch, out_ch, use_spectral_norm=False):
    n = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
    if use_spectral_norm:
        nn.init.orthogonal_(n.weight.data)
        n = spectral_norm(n)
    else:
        nn.init.normal_(n.weight.data, 0.0, 0.02)
    return n


class ConditioningAugmentor(nn.Module):
    def __init__(self, in_dim, out_dim, use_spectral_norm=False):
        super().__init__()
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, out_dim * 2)
        if use_spectral_norm:
            nn.init.orthogonal_(self.fc.weight.data)
            nn.init.constant_(self.fc.bias.data, 0.0)
            self.fc = spectral_norm(self.fc)
        else:
            nn.init.normal_(self.fc.weight.data, 0.0, 0.02)
            nn.init.constant_(self.fc.bias.data, 0.0)

    def _encode(self, x):
        x = self.fc(x)
        mu = x[:, :self.out_dim]
        logvar = x[:, self.out_dim:]
        return mu, logvar

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        x = mu + std * eps
        return x

    def forward(self, x):
        mu, logvar = self._encode(x)
        x = self._reparameterize(mu, logvar)
        return x, mu, logvar


class ResBlockUp(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        condition_dim=768,
        norm="bn",
        use_upsample=True,
        use_spectral_norm=False,
    ):
        super().__init__()

        # residual
        self.conv1 = conv3x3(in_channels, out_channels, use_spectral_norm)
        self.conv2 = conv3x3(out_channels, out_channels, use_spectral_norm)

        self.use_norm = (norm != "none")
        if self.use_norm:
            if norm == "bn":
                self.norm1 = ConditionalBatchNorm2d(
                    in_channels, condition_dim)
                self.norm2 = ConditionalBatchNorm2d(
                    out_channels, condition_dim)
            elif norm == "in":
                self.norm1 = AdaptiveInstanceNorm2d(
                    in_channels, condition_dim)
                self.norm2 = AdaptiveInstanceNorm2d(
                    out_channels, condition_dim)
            else:
                raise ValueError

        # bypass
        self.use_conv_bypass = in_channels != out_channels
        if self.use_conv_bypass:
            self.conv_bypass = conv1x1(
                in_channels, out_channels, use_spectral_norm)

        # upsample
        self.use_upsample = use_upsample
        if self.use_upsample:
            self.upsampler = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, y=None):
        x_bypass = x

        # residual
        if self.use_norm:
            x = self.norm1(x, y)
        x = F.relu(x)
        if self.use_upsample:
            x = self.upsampler(x)
        x = self.conv1(x)
        if self.use_norm:
            x = self.norm2(x, y)
        x = F.relu(x)
        x = self.conv2(x)

        # bypass
        if self.use_upsample:
            x_bypass = self.upsampler(x_bypass)
        if self.use_conv_bypass:
            x_bypass = self.conv_bypass(x_bypass)

        x = x + x_bypass
        return x


class ResBlockDown(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm="none",
        use_downsample=True,
        use_spectral_norm=True,
        is_first_block=False,
    ):
        super().__init__()

        # residual
        self.conv1 = conv3x3(in_channels, out_channels, use_spectral_norm)
        self.conv2 = conv3x3(out_channels, out_channels, use_spectral_norm)

        self.is_first_block = is_first_block
        self.use_norm = (norm != "none")
        if self.use_norm:
            if norm == "bn":
                if not self.is_first_block:
                    self.norm1 = BatchNorm2d(in_channels)
                self.norm2 = BatchNorm2d(out_channels)
            elif norm == "in":
                if not self.is_first_block:
                    self.norm1 = InstanceNorm2d(in_channels)
                self.norm2 = InstanceNorm2d(out_channels)
            else:
                raise ValueError

        # bypass
        self.use_conv_bypass = in_channels != out_channels
        if self.use_conv_bypass:
            self.conv_bypass = conv1x1(
                in_channels, out_channels, use_spectral_norm)

        # downsample
        self.use_downsample = use_downsample
        if self.use_downsample:
            self.downsampler = nn.AvgPool2d(2)

    def forward(self, x):
        x_bypass = x

        # residual
        if not self.is_first_block:
            if self.use_norm:
                x = self.norm1(x)
            x = F.relu(x)
        x = self.conv1(x)
        if self.use_norm:
            x = self.norm2(x)
        x = F.relu(x)
        x = self.conv2(x)
        if self.use_downsample:
            x = self.downsampler(x)

        # bypass
        if self.use_conv_bypass:
            if self.is_first_block:
                if self.use_downsample:
                    x_bypass = self.downsampler(x_bypass)
                x_bypass = self.conv_bypass(x_bypass)
            else:
                x_bypass = self.conv_bypass(x_bypass)
                if self.use_downsample:
                    x_bypass = self.downsampler(x_bypass)
        else:
            if self.use_downsample:
                x_bypass = self.downsampler(x_bypass)

        x = x + x_bypass
        return x
