import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .activations import ACTIVATIONS


def conv1x1(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, 1, bias=False)


def conv3x3(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)


class SelfAttention(nn.Module):
    def __init__(self, channels, use_spectral_norm=True):
        """
        Parameters
        ----------
        channels : int
        use_spectral_norm : bool
            by default True
        """
        super().__init__()
        self.f_x = nn.Conv2d(channels, channels // 8, 1)
        self.g_x = nn.Conv2d(channels, channels // 8, 1)
        self.h_x = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        if use_spectral_norm:
            self.f_x = spectral_norm(self.f_x)
            self.g_x = spectral_norm(self.g_x)
            self.h_x = spectral_norm(self.h_x)

    def forward(self, x):
        """Return image feature map through self-attention block

        Parameters
        ----------
        x : torch.tensor
            shape (B, C, H, W)

        Returns
        -------
        y : torch.tensor
            shape (B, C, H, W) (same shape as input "x")
        """
        b, c, h, w = x.size()
        n = h * w

        fx = self.f_x(x).view(b, c // 8, n)
        gx = self.g_x(x).view(b, c // 8, n)
        sx = torch.bmm(fx.permute(0, 2, 1), gx)  # f(x)^{T} * g(x)
        beta = F.softmax(sx, dim=1)

        hx = self.h_x(x).view(b, c, n)
        ox = torch.bmm(hx, beta).view(b, c, h, w)

        y = self.gamma * ox + x
        return y


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, in_channels, condition_dim):
        """
        Parameters
        ----------
        in_channels : int
            Channel size of input image feature.
        condition_dim : int
            Dimension of input conditional feature (B, condition_dim).
        """
        super().__init__()
        self.channels = in_channels
        self.bn = nn.BatchNorm2d(in_channels, affine=False)
        self.fc = nn.Linear(condition_dim, in_channels * 2)
        self.fc.weight.data[:, :in_channels] = 1
        self.fc.weight.data[:, in_channels:] = 0

    def forward(self, activations, condition):
        """Return batchnormed feature map with conditional affine transform.

        Parameters
        ----------
        activations : torch.tensor
            shape (B, C, H, W).
        condition : torch.tensor
            shape (B, D).
            It becomes channel-wise gamma and beta of affine transform.

        Returns
        -------
        activations : torch.tensor
            shape (B, C, H, W) (same shape as input "activations")
            gamma * BN(activation) + beta
        """
        condition = self.fc(condition)
        gamma = condition[:, :self.channels].unsqueeze(2).unsqueeze(3)
        beta = condition[:, self.channels:].unsqueeze(2).unsqueeze(3)
        activations = self.bn(activations)
        return activations.mul(gamma).add(beta)


class ConditioningAugmentor(nn.Module):
    """
    Conditioning Augmentation helps the manifold
    turn out more smooth by sampling from a Gaussian
    where mean and variance are functions of the text
    embedding.
    """

    def __init__(self, emb_dim, ca_dim):
        """
        Parameters
        ----------
        emb_dim : int
            Dimension of input feature vector.
        ca_dim : int
            Dimension of output feature vector.
        """
        super().__init__()
        self.t_dim = emb_dim
        self.c_dim = ca_dim
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def _encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def _reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        """Return re-parametarized text_embedding (like VAE).

        Parameters
        ----------
        text_embedding : torch.tensor
            shape (B, D=emb_dim)
            In GeNeVA paper, it's denoted as "h_t"
            before Conditioning Augmentation.

        Returns
        -------
        c_code : torch.tensor
            shape (B, D=ca_dim)
            Re-parametarized text_embedding.
        mu : torch.tensor
            shape (B, D=ca_dim)
            Stats "mu" for re-parametarization.
        logvar : torch.tensor
            shape (B, D=ca_dim)
            Stats log("var") for re-parametarization.
        """
        mu, logvar = self._encode(text_embedding)
        c_code = self._reparametrize(mu, logvar)
        return c_code, mu, logvar


class ResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim,
                 conditional=True, use_self_attn=False,
                 use_spectral_norm=True, activation="relu"):
        """
        Parameters
        ----------
        in_channels : int
            Channel size of input image feature
        out_channels : int
            Channel size of output image feature
        embedding_dim : int
            Dimension of input conditioning text feature
            for ConditionalBatchNorm2d.
        conditional : bool, optional
            If True, use ConditionalBatchNorm2d instead of BatchNorm2d, by default True
        use_self_attn : bool, optional
            If True, append Self-Attention Block at the end of residual pass, by default False
        use_spectral_norm : bool, optional
            If True, enable spectral normalization for all learnable weights, by default True
        activation : str, optional
            "relu", "leaky_relu", or "selu", by default "relu"
        """
        super().__init__()
        self.conditional = conditional

        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.conv_shortcut = conv1x1(in_channels, out_channels)
        self.upsampler = nn.Upsample(scale_factor=2, mode="nearest")
        self.activation = ACTIVATIONS[activation]

        if conditional:
            self.bn1 = ConditionalBatchNorm2d(in_channels, embedding_dim)
            self.bn2 = ConditionalBatchNorm2d(out_channels, embedding_dim)
        else:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            self.conv_shortcut = spectral_norm(self.conv_shortcut)

        self.use_self_attn = use_self_attn
        if self.use_self_attn:
            self.self_attention = SelfAttention(
                out_channels, use_spectral_norm)

    def forward(self, x, y=None):
        """Return H(x) = x + F(x), Upsample(2) applied.

        Parameters
        ----------
        x : torch.tensor
            shape (B, C_in, H, W)

        Returns
        -------
        y : torch.tensor
            shape (B, C_out, H * 2, W * 2)
        """
        x_bypass = x

        # residual pass
        x = self.bn1(x, y) if self.conditional else self.bn1(x)
        x = self.activation(x)
        x = self.upsampler(x)
        x = self.conv1(x)
        x = self.bn2(x, y) if self.conditional else self.bn2(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.use_self_attn:
            x = self.self_attention(x)

        # bypass
        x_bypass = self.upsampler(x_bypass)
        x_bypass = self.conv_shortcut(x_bypass)

        return x + x_bypass


class ResDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True,
                 first_block=False, activation="relu",
                 use_self_attn=False, use_spectral_norm=True):
        """
        Parameters
        ----------
        in_channels : int
            Channel size of input image feature
        out_channels : int
            Channel size of output image feature
        downsample : bool, optional
            If True, apply AvgPool2d(2) to image feature, by default True
        first_block : bool, optional
            If True, skip first activation func in residual pass, by default False
        activation : str, optional
            "relu", "leaky_relu", or "selu", by default "relu"
        use_self_attn : bool, optional
            If True, append Self-Attention Block at the end of residual pass, by default False
        use_spectral_norm : bool, optional
            If True, enable spectral normalization for all learnable weights, by default True
        """
        super().__init__()
        self.first_block = first_block

        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.conv_shortcut = conv1x1(in_channels, out_channels)
        self.downsample = downsample
        self.downsampler = nn.AvgPool2d(kernel_size=2)
        self.activation = ACTIVATIONS[activation]

        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            self.conv_shortcut = spectral_norm(self.conv_shortcut)

        self.use_self_attn = use_self_attn
        if self.use_self_attn:
            self.self_attention = SelfAttention(
                out_channels, use_spectral_norm)

    def forward(self, x):
        """Return H(x) = x + F(x)

        Parameters
        ----------
        x : torch.tensor
            shape (B, C_in, H, W)

        Returns
        -------
        y : torch.tensor
            shape (B, C_out, H', W')
            If downsample == True, H', W' are half of H, W.
        """
        x_bypass = x

        # residual pass
        if not self.first_block:
            x = self.activation(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.downsample:
            x = self.downsampler(x)
        if self.use_self_attn:
            x = self.self_attention(x)

        # bypass
        if self.downsample:
            if self.first_block:
                x_bypass = self.downsampler(x_bypass)
                x_bypass = self.conv_shortcut(x_bypass)
            else:
                x_bypass = self.conv_shortcut(x_bypass)
                x_bypass = self.downsampler(x_bypass)

        return x + x_bypass
