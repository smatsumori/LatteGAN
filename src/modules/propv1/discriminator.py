import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ResBlockDown, conv1x1


class Discriminator(nn.Module):
    def __init__(
        self,
        condition_dim=768,
        discriminator_sn=True,
        aux_detection_dim=58,
        fusion="subtract",
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
        self.projector = conv1x1(condition_dim, 1024)

        # auxiliary object detector
        self.aux_detector = nn.Sequential(
            conv1x1(1024, 256, discriminator_sn),
            nn.ReLU(),
            conv1x1(256, aux_detection_dim, discriminator_sn),
        )

    def forward(self, x_src, x_tgt, y):
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
        x = torch.sum(x, dim=(2, 3), keepdim=True)

        # adversarial
        adv = self.linear(x)

        # projection
        y = y.unsqueeze(2).unsqueeze(3)
        y = self.projector(y)
        y = torch.sum(x * y, dim=1, keepdim=True)

        adv = (adv + y).squeeze(3).squeeze(2)

        # auxiliary object detector
        aux = self.aux_detector(x)
        aux = aux.squeeze(3).squeeze(2)

        return adv, aux
