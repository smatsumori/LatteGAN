import torch
import torch.nn as nn

from .layers import ResUpBlock, ConditioningAugmentor, conv3x3
from .activations import ACTIVATIONS


class GeneratorRecurrentGANRes(nn.Module):
    def __init__(self, conditional=True, condition_dim=1024,
                 conditioning_dim=256, noise_dim=100, generator_sn=False,
                 activation="leaky_relu", gen_fusion="concat",
                 self_attention=False, cond_kl_reg=1):
        """
        Parameters
        ----------
        conditional : bool, optional
            If True, use ConditionalBatchNorm2d(x, h_t), by default True
        condition_dim : int, optional
            Dimension of input conditional text embedding "h_t", by default 1024
        conditioning_dim : int, optional
            Dimension of vector from ConditinalAugmentation, by default 256
        noise_dim : int, optional
            Dimension of noise vector~N(0,I), by default 100
        generator_sn : bool, optional
            If True, apply spectral normalization, by default False
        activation : str, optional
            "relu", "leaky_relu", or "selu", by default "leaky_relu"
        gen_fusion : str, optional
            Merge "f_{G_{t-1}}" by "concat" or "gate", by default "concat".
            Channel size of "f_{G_{t-1}}" MUST BE 512 (default).
            If "concat", "f_{G_{t-1}}" is concatenated along channels with middle-layer image feature.
            If "gate", calcurate channel-wise sigmoid gate from "h_t", and merge feature
            calculated by `gate * x + (1 - gate) * "f_{G_{t-1}}"`.
        self_attention : bool, optional
            If True, use SelfAttention at (512, 16, 16) residual layer, by default False
        cond_kl_reg : int, optional
            Weight of ConditionalAugmentation KL divergence loss, by default 1.
            If cond_kl_reg == 0, skip ConditionalAugmentation.
        """
        super().__init__()
        self.noise_dim = noise_dim
        self.z_dim = noise_dim + conditioning_dim

        self.fc1 = nn.Linear(self.z_dim, 1024 * 4 * 4)
        # state size. (1024) x 4 x 4
        self.resup1 = ResUpBlock(1024, 1024, condition_dim,
                                 conditional=conditional,
                                 use_self_attn=False,
                                 use_spectral_norm=generator_sn,
                                 activation=activation)
        # state size. (1024) x 8 x 8
        self.resup2 = ResUpBlock(1024, 512, condition_dim,
                                 conditional=conditional,
                                 use_self_attn=False,
                                 use_spectral_norm=generator_sn,
                                 activation=activation)

        assert gen_fusion in ["gate", "concat"]
        self.gen_fusion = gen_fusion
        if self.gen_fusion == "concat":
            extra_channels = 512
        elif self.gen_fusion == "gate":
            extra_channels = 0
            self.gate = nn.Sequential(
                nn.Linear(condition_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.Sigmoid()
            )

        # state size. (512) x 16 x 16
        self.resup3 = ResUpBlock(512 + extra_channels, 256, condition_dim,
                                 conditional=conditional,
                                 use_self_attn=self_attention,
                                 use_spectral_norm=generator_sn,
                                 activation=activation)
        # state size. (256) x 32 x 32
        self.resup4 = ResUpBlock(256, 128, condition_dim,
                                 conditional=conditional,
                                 use_self_attn=False,
                                 use_spectral_norm=generator_sn,
                                 activation=activation)
        # state size. (128) x 64 x 64
        self.resup5 = ResUpBlock(128, 64, condition_dim,
                                 conditional=conditional,
                                 use_self_attn=False,
                                 use_spectral_norm=generator_sn,
                                 activation=activation)
        # state size. (64) x 128 x 128
        self.bn = nn.BatchNorm2d(64)
        self.activation = ACTIVATIONS[activation]
        self.conv = conv3x3(64, 3)
        # state size. (3) x 128 x 128
        self.tanh = nn.Tanh()

        self.cond_kl_reg = cond_kl_reg
        if self.cond_kl_reg is not None:
            self.condition_projector = ConditioningAugmentor(
                condition_dim, conditioning_dim)
        else:
            self.condition_projector = nn.Linear(
                condition_dim, conditioning_dim)

    def forward(self, z, y, img_feats):
        """Generate fake images from noise, "h_t", and "f_G_{t-1}".

        Parameters
        ----------
        z : torch.tensor
            shape (B, Nz)
            Nz = noise_dim (default 100).
            Gaussian noise for generator input.
        y : torch.tensor
            shape (B, Nc)
            Nc = condition_dim (default 1024).
            Text feature from (GRU in paper) ConditionEncoder, so-called "h_t".
        img_feats : torch.tensor
            shape (B, C=512, H=16, W=16)
            Image feature map from ImageEncoder, so-called  "f_G_{t-1}".
            Shape is fixed by ImageEncoder: "x_{t-1}" (B, 3, 128, 128) --> "f_G_{t-1}" (B, 512, 16, 16).

        Returns
        -------
        x : torch.tensor
            shape (B, 3, 128, 128)
            Generated fake images, value range = (-1, 1).
        mu : torch.tensor or None
            shape (B, C).
            C = conditioning_dim (default 256).
            Mu of conditioning augmentation if cond_kl_reg is specified.
        logvar: torch.tensor or None
            shape (B, C).
            C = conditioning_dim (default 256).
            Log of variance of conditioning augmentation if cond_kl_reg is specified.
        sigma: torch.tensor or None
            shape (B, 512, 1, 1).
            Channel-wise gate for both image feature and text feature if gen_fusion == gate.
        """
        mu, logvar = None, None
        cond_y = self.condition_projector(y)
        if self.cond_kl_reg is not None:
            cond_y, mu, logvar = cond_y

        z = torch.cat([z, cond_y], dim=1)

        x = self.fc1(z)
        x = x.view(-1, 1024, 4, 4)
        x = self.resup1(x, y)
        x = self.resup2(x, y)

        if self.gen_fusion == "concat":
            sigma = None
            x = torch.cat([x, img_feats], dim=1)
        elif self.gen_fusion == "gate":
            sigma = self.gate(y)
            sigma = sigma.unsqueeze(2).unsqueeze(3)
            x = x * sigma + img_feats * (1 - sigma)

        x = self.resup3(x, y)
        x = self.resup4(x, y)
        x = self.resup5(x, y)
        x = self.bn(x)
        x = self.activation(x)
        x = self.conv(x)
        x = self.tanh(x)

        return x, mu, logvar, sigma
