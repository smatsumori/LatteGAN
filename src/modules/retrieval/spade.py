import torch.nn as nn


class SPADE(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
    ):
        super().__init__()

        self.conv_gamma = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding)
        self.conv_beta = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels, affine=False)

    def forward(self, x, y):
        gamma = self.conv_gamma(y)
        beta = self.conv_beta(y)
        x = self.norm(x)
        x = gamma * x + beta
        return x
