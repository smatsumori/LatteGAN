import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .normalization import BatchNorm2d, InstanceNorm2d


class SPADE(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        norm="bn",
        use_specral_norm=False,
    ):
        super().__init__()

        # norm1: normalize semantic feature map
        # norm2: normalize image feature map
        if norm == "bn":
            self.norm1 = BatchNorm2d(in_channels)
            self.norm2 = nn.BatchNorm2d(out_channels, affine=False)
        elif norm == "in":
            self.norm1 = InstanceNorm2d(in_channels)
            self.norm2 = InstanceNorm2d(out_channels)
        else:
            raise ValueError

        # convolution after relu, for semantic feature map
        # in_channels --> out_channels
        # out_channels is equal to image feature map
        self.conv_gamma = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding)
        self.conv_beta = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding)

        if use_specral_norm:
            nn.init.orthogonal_(self.conv_gamma.weight.data)
            nn.init.orthogonal_(self.conv_beta.weight.data)
            self.conv_gamma = spectral_norm(self.conv_gamma)
            self.conv_beta = spectral_norm(self.conv_beta)
        else:
            nn.init.normal_(self.conv_gamma.weight.data, 0.0, 0.02)
            nn.init.normal_(self.conv_beta.weight.data, 0.0, 0.02)

    def forward(self, x, y):
        """SPADE of pre-bn-resblock-out

        Args:
            x (torch.Tensor): image feature map, shape=(B, out_channels, H, W).
            y (torch.Tensor): semantic feature map, shape=(B, in_channels, H, W).

        Returns:
            torch.Tensor: x applied SPADE operation.
        """
        y = self.norm1(y)
        y = F.relu(y)
        gamma = self.conv_gamma(y)
        beta = self.conv_beta(y)

        x = self.norm2(x)
        x = gamma * x + beta
        return x
