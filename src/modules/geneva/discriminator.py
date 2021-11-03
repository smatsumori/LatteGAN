import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from .layers import ResDownBlock
from .activations import ACTIVATIONS


class DiscriminatorAdditiveGANRes(nn.Module):
    def __init__(self, activation="leaky_relu", disc_sn=True, disc_img_conditioning="subtract",
                 conditioning="projection", disc_cond_channels=512, self_attention=False,
                 condition_dim=1024, num_objects=24):
        """
        Parameters
        ----------
        activation : str, optional
            "relu", "leaky_relu", or "selu", by default "leaky_relu"
        disc_sn : bool, optional
            If True, apply spectral normalization, by default True
        disc_img_conditioning : str, optional
            "concat" or "subtract", if "concat", E_D(x_{t-1}) and E_D(x_t)
            are fused by channel-side concat way and feeded D(.),
            if "subtact", execute E_D(x_t) - E_D(x_{t-1}), by default "subtract"
        conditioning : str, optional
            "concat" or "projection". If "concat", "h_t" is broadcasted along hw size,
            and concatenated with fused image feature map. If "projection", "h_t" is used
            like prototype class vector in projection discriminator. by default "projection"
        disc_cond_channels : int, optional
            Dimension of vector from "h_t", it is valid
            when conditioning="concat" specified, by default 512
        self_attention : bool, optional
            If True, use SelfAttention at (256, 16, 16) residual layer, by default False
        condition_dim : int, optional
            Dimension of input text embedding "h_t", by default 1024
        num_objects : int, optional
            Number of possible combination (color, shape), by default 24
        """
        super().__init__()
        self.activation = ACTIVATIONS[activation]

        self.resdown1 = ResDownBlock(3, 64, downsample=True, first_block=True,
                                     activation=activation, use_self_attn=False,
                                     use_spectral_norm=disc_sn)
        # state size. (64) x 64 x 64
        self.resdown2 = ResDownBlock(64, 128, downsample=True,
                                     activation=activation, use_self_attn=False,
                                     use_spectral_norm=disc_sn)
        # state size. (128) x 32 x 32
        self.resdown3 = ResDownBlock(128, 256, downsample=True,
                                     activation=activation, use_self_attn=False,
                                     use_spectral_norm=disc_sn)

        extra_channels = 0
        assert disc_img_conditioning in ["concat", "subtract"]
        self.disc_img_conditioning = disc_img_conditioning
        if self.disc_img_conditioning == "concat":
            extra_channels += 256
        if conditioning == "concat":
            extra_channels += disc_cond_channels

        # state size. (256) x 16 x 16
        self.resdown4 = ResDownBlock(256 + extra_channels, 512, downsample=True,
                                     activation=activation, use_self_attn=self_attention,
                                     use_spectral_norm=disc_sn)
        # state size. (512) x 8 x 8
        self.resdown5 = ResDownBlock(512, 1024, downsample=True,
                                     activation=activation, use_self_attn=False,
                                     use_spectral_norm=disc_sn)
        # state size. (1024) x 4 x 4
        self.resdown6 = ResDownBlock(1024, 1024, downsample=False,
                                     activation=activation, use_self_attn=False,
                                     use_spectral_norm=disc_sn)

        self.linear = nn.Linear(1024, 1)
        if disc_sn:
            self.linear = spectral_norm(self.linear)

        assert conditioning in ["projection", "concat"]
        self.conditioning = conditioning
        if self.conditioning == "projection":
            self.condition_projector = nn.Sequential(
                nn.Linear(condition_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
            )
        elif self.conditioning == "concat":
            self.condition_projector = nn.Sequential(
                nn.Linear(condition_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, disc_cond_channels),
            )

        self.aux_objective = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_objects),
        )

    def forward(self, x, y, prev_image):
        """Discriminate real or fake conpared with prev_image, and aux task.

        Parameters
        ----------
        x : torch.tensor
            shape (B, C=3, H=128, W=128)
            Real or fake image. Value range = (-1, 1).
            It's represented by "x_t" or "x~_t" in GeNeVA paper.
        y : torch.tensor
            shape (B, C)
            Text embedded feature, so-called "h_t" in GeNeVA paper.
        prev_image : torch.tensor
            shape (B, C=3, H=128, W=128)
            Real src image, so-called "x_{t-1}" in GeNeVA paper.

        Returns
        -------
        out : torch.tensor
            shape (B,)
            Real or fake regression (it means raw output of nn.Linear).
        aux : torch.tensor
            shape (B, N)
            N = num_objects (default 24).
            It is used for auxialiary task, binary cross entropy over
            all the N possible objects at that time step.
            Binary label for each object indicating whether it is present
            in the scene at the current time step.
        intermediate_features : torch.tensor
            shape (B, C=1024, H=4, W=4)
            Final hidden state in convolution module of discriminator.
        """
        prev_image = self.resdown1(prev_image)
        prev_image = self.resdown2(prev_image)
        prev_image = self.resdown3(prev_image)

        y = self.condition_projector(y)

        x = self.resdown1(x)
        x = self.resdown2(x)
        x = self.resdown3(x)

        if self.disc_img_conditioning == "concat":
            x = torch.cat((x, prev_image), dim=1)
        else:
            x = x - prev_image

        if self.conditioning == "concat":
            y = y.unsqueeze(2).unsqueeze(3)
            y = y.repeat(1, 1, x.size(2), x.size(3))
            x = torch.cat([x, y], dim=1)

        x = self.resdown4(x)
        x = self.resdown5(x)
        x = self.resdown6(x)

        intermediate_features = x

        x = self.activation(x)
        x = torch.sum(x, dim=(2, 3))  # Global Sum Pooling

        # real or fake "logit"
        out = self.linear(x).squeeze(1)
        if self.conditioning == "projection":
            c = torch.sum(y * x, dim=1)
            out = out + c

        # object binary classification "logit"
        aux = torch.sigmoid(self.aux_objective(x))

        return out, aux, intermediate_features
