import torch.nn as nn

from .layers import ResBlockDown


# TODO: multi-scale
# TODO: variable image size
class ImageEncoder(nn.Module):
    def __init__(
        self,
        image_feat_dim=512,
        norm="bn",
        use_spectral_norm=False,
    ):
        super().__init__()
        self.image_encoder = nn.Sequential(
            # --> 64 x 64 x 64
            ResBlockDown(
                3,
                64,
                norm=norm,
                use_spectral_norm=use_spectral_norm,
                is_first_block=True,
            ),
            # --> 128 x 32 x 32
            ResBlockDown(
                64,
                128,
                norm=norm,
                use_spectral_norm=use_spectral_norm,
            ),
            # --> image_feat_dim x 16 x 16
            ResBlockDown(
                128,
                image_feat_dim,
                norm=norm,
                use_spectral_norm=use_spectral_norm,
            ),
        )

    def forward(self, x):
        x = self.image_encoder(x)
        return x


class ImageEncoderMultiScale(nn.Module):
    def __init__(
        self,
        depth=4,  # downscale image by 2^(depth)
        image_feat_dim=512,
        norm="bn",
        use_spectral_norm=False,
    ):
        super().__init__()

        assert depth >= 1
        assert 64 <= image_feat_dim

        base = 64
        down_modules = []

        # first downsample block
        _resdown = ResBlockDown(
            in_channels=3,
            out_channels=min(base, image_feat_dim),
            norm=norm,
            use_spectral_norm=use_spectral_norm,
            is_first_block=True,
        )
        base = min(base, image_feat_dim)
        down_modules.append(_resdown)

        # append downsample block
        for _ in range(depth - 1):
            _resdown = ResBlockDown(
                in_channels=base,
                out_channels=min(base * 2, image_feat_dim),
                norm=norm,
                use_spectral_norm=use_spectral_norm,
            )
            base = min(base * 2, image_feat_dim)
            down_modules.append(_resdown)

        self.resdownblocks = nn.ModuleList(down_modules)
        assert base == image_feat_dim

    def forward(self, x):
        x_multiscales = []

        for n in self.resdownblocks:
            x = n(x)
            x_multiscales.append(x)

        return x_multiscales
