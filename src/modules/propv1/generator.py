import torch
import torch.nn as nn

from .layers import ResBlockUp, ConditioningAugmentor, conv1x1, conv3x3
from .normalization import BatchNorm2d, InstanceNorm2d
from .tirg import TIRG, TIRGSPADE
from .tidg import TIDG


class Generator(nn.Module):
    def __init__(
        self,
        condition_dim=768,
        condaug_out_dim=256,
        cond_kl_reg=1.0,
        noise_dim=100,
        norm="bn",
        generator_sn=False,
        fusion="concat",
        image_feat_dim=512,
        sta="none",
        nhead=1,
        res_mask_post=False,
        multi_channel_gate=False,
        use_relnet=False,
    ):
        super().__init__()

        # conditioning augmentor
        self.cond_kl_reg = cond_kl_reg
        if self.cond_kl_reg > 0.0:
            self.projector = ConditioningAugmentor(
                condition_dim,
                condaug_out_dim,
                generator_sn,
            )
        else:
            self.projector = conv1x1(
                condition_dim,
                condaug_out_dim,
                generator_sn,
            )

        # generator from text
        self.fc = conv1x1(
            condaug_out_dim + noise_dim,
            1024 * 4 * 4,
            generator_sn,
        )
        self.resup1 = ResBlockUp(
            1024,
            1024,
            condition_dim,
            norm=norm,
            use_spectral_norm=generator_sn,
        )
        self.resup2 = ResBlockUp(
            1024,
            512,
            condition_dim,
            norm=norm,
            use_spectral_norm=generator_sn,
        )

        # fusion module
        self.fusion = fusion
        if self.fusion == "concat":
            # text(512) + image
            in_channels = 512 + image_feat_dim
        elif self.fusion == "tirg":
            # image(512)
            in_channels = image_feat_dim
            self.tirg = TIRG(
                image_feat_dim,
                text_dim=512,
                hidden_dim=512,
                norm=norm,
                sta=sta,
                nhead=nhead,
                memory_dim=condition_dim,
                res_mask_post=res_mask_post,
                multi_channel_gate=multi_channel_gate,
                use_spectral_norm=generator_sn,
                use_relnet=use_relnet,
            )
        elif self.fusion == "tirg-spade":
            # image(512)
            in_channels = image_feat_dim
            self.tirg_spade = TIRGSPADE(
                image_feat_dim,
                text_dim=512,
                hidden_dim=512,
                norm=norm,
                sta=sta,
                nhead=nhead,
                memory_dim=condition_dim,
                res_mask_post=res_mask_post,
                multi_channel_gate=multi_channel_gate,
                use_spectral_norm=generator_sn,
                use_relnet=use_relnet,
            )
        else:
            raise ValueError

        # generator to output image
        self.resup3 = ResBlockUp(
            in_channels,
            256,
            condition_dim,
            norm=norm,
            use_spectral_norm=generator_sn,
        )
        self.resup4 = ResBlockUp(
            256,
            128,
            condition_dim,
            norm=norm,
            use_spectral_norm=generator_sn,
        )
        self.resup5 = ResBlockUp(
            128,
            64,
            condition_dim,
            norm=norm,
            use_spectral_norm=generator_sn,
        )
        if norm == "bn":
            self.norm = BatchNorm2d(64)
        elif norm == "in":
            self.norm = InstanceNorm2d(64)
        else:
            raise ValueError
        self.relu = nn.ReLU()
        self.conv = conv3x3(64, 3, generator_sn)
        self.tanh = nn.Tanh()

    def forward(
        self,
        z,
        y,
        img_feat,
        memories=None,
        lengths=None,
    ):
        # conditioning augmentor
        mu, logvar = None, None
        if self.cond_kl_reg > 0.0:
            cond_y, mu, logvar = self.projector(y)
            cond_y = cond_y.unsqueeze(2).unsqueeze(3)
        else:
            cond_y = y.unsqueeze(2).unsqueeze(3)
            cond_y = self.projector(y)

        z = z.unsqueeze(2).unsqueeze(3)
        z = torch.cat([z, cond_y], dim=1)

        # generator from text
        x = self.fc(z)
        x = x.view(-1, 1024, 4, 4)
        x = self.resup1(x, y)
        x = self.resup2(x, y)

        # fusion module
        if self.fusion == "concat":
            gate = None
            x = torch.cat([x, img_feat], dim=1)
        elif self.fusion == "tirg":
            x, gate = self.tirg(
                img_feat,
                x,
                memories,
                lengths,
            )
        elif self.fusion == "tirg-spade":
            x, gate = self.tirg_spade(
                img_feat,
                x,
                memories,
                lengths,
            )

        # generator to output image
        x = self.resup3(x, y)
        x = self.resup4(x, y)
        x = self.resup5(x, y)
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.tanh(x)

        return x, mu, logvar, gate


class GeneratorMultiScale(nn.Module):
    def __init__(
        self,
        condition_dim=768,
        condaug_out_dim=256,
        cond_kl_reg=1.0,
        noise_dim=100,
        norm="bn",
        generator_sn=False,
        fusion="concat",
        fusion_image_size=8,
        output_image_size=128,
        image_feat_dim=512,
        sta="none",
        nhead=1,
        res_mask_post=False,
        multi_channel_gate=False,
    ):
        super().__init__()

        # conditioning augmentor
        self.cond_kl_reg = cond_kl_reg
        if self.cond_kl_reg > 0.0:
            self.projector = ConditioningAugmentor(
                condition_dim,
                condaug_out_dim,
                generator_sn,
            )
        else:
            self.projector = conv1x1(
                condition_dim,
                condaug_out_dim,
                generator_sn,
            )

        # generator from text
        assert fusion_image_size >= 8
        self.fc = conv1x1(
            condaug_out_dim + noise_dim,
            1024 * 4 * 4,
            generator_sn,
        )
        # output.shape = (B, 512, 8, 8)
        self.resup1 = ResBlockUp(
            1024,
            512,
            condition_dim,
            norm=norm,
            use_spectral_norm=generator_sn,
        )

        # upsample to match fusion image size
        # e.g.)
        #    (B, 512, 8, 8)
        #    (B, 256, 16, 16)
        imsize = 8
        base = 512
        text_up_modules = []
        while imsize < fusion_image_size:
            _resup = ResBlockUp(
                base,
                base // 2,
                condition_dim,
                norm=norm,
                use_spectral_norm=generator_sn,
            )
            imsize *= 2
            base //= 2
            text_up_modules.append(_resup)
        assert imsize == fusion_image_size
        self.text_resups = None
        if len(text_up_modules) > 0:
            self.text_resups = nn.ModuleList(text_up_modules)

        # fusion module
        self.fusion = fusion
        if self.fusion == "concat":
            # text + image
            in_channels = base + image_feat_dim
        elif self.fusion == "tirg":
            # image
            in_channels = image_feat_dim
            self.tirg = TIRG(
                image_feat_dim,
                text_dim=base,
                hidden_dim=base,
                norm=norm,
                sta=sta,
                nhead=nhead,
                memory_dim=condition_dim,
                res_mask_post=res_mask_post,
                multi_channel_gate=multi_channel_gate,
                use_spectral_norm=generator_sn,
            )
        else:
            raise ValueError

        # after fusion, multi-scale by TIDG
        # ResBlockUp: ht^{d} --> ht^{d-1}
        # TIDG: hs^{d-1}, ht^{d-1}, y --> ht^{d-1}
        out_channels = base // 2
        up_modules = []
        tidg_modules = []
        while imsize < (output_image_size // 2):
            _resup = ResBlockUp(
                in_channels,
                out_channels,
                condition_dim,
                norm=norm,
                use_spectral_norm=generator_sn,
            )
            imsize *= 2
            in_channels = out_channels
            out_channels //= 2
            up_modules.append(_resup)

            _tidg = TIDG(
                in_channels,
                condition_dim,
                in_channels,
                norm=norm,
                res_mask_post=res_mask_post,
                multi_channel_gate=multi_channel_gate,
                use_spectral_norm=generator_sn,
            )
            tidg_modules.append(_tidg)

        assert imsize == (output_image_size // 2)
        self.resups = nn.ModuleList(up_modules)
        self.tidgs = nn.ModuleList(tidg_modules)

        # feature to image
        self.last_resup = ResBlockUp(
            in_channels,
            out_channels,
            condition_dim,
            norm=norm,
            use_spectral_norm=generator_sn,
        )
        if norm == "bn":
            self.norm = BatchNorm2d(out_channels)
        elif norm == "in":
            self.norm = InstanceNorm2d(out_channels)
        else:
            raise ValueError
        self.relu = nn.ReLU()
        self.conv = conv3x3(out_channels, 3, generator_sn)
        self.tanh = nn.Tanh()

    def forward(
        self,
        z,
        y,
        img_feats,
        memories=None,
        lengths=None,
    ):
        img_feats = img_feats[::-1]

        # conditioning augmentor
        mu, logvar = None, None
        if self.cond_kl_reg > 0.0:
            cond_y, mu, logvar = self.projector(y)
            cond_y = cond_y.unsqueeze(2).unsqueeze(3)
        else:
            cond_y = y.unsqueeze(2).unsqueeze(3)
            cond_y = self.projector(y)

        z = z.unsqueeze(2).unsqueeze(3)
        z = torch.cat([z, cond_y], dim=1)

        # generator from text
        x = self.fc(z)
        x = x.view(-1, 1024, 4, 4)
        x = self.resup1(x, y)
        if self.text_resups is not None:
            for n in self.text_resups:
                x = n(x, y)

        # fusion module
        if self.fusion == "concat":
            x = torch.cat([x, img_feats[0]], dim=1)
        elif self.fusion == "tirg":
            x, _ = self.tirg(
                img_feats[0],
                x,
                memories,
                lengths,
            )

        # generator to output image
        for i in range(len(self.resups)):
            x = self.resups[i](x, y)
            x, _ = self.tidgs[i](
                img_feats[i + 1],
                x,
                y,
            )

        x = self.last_resup(x, y)
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.tanh(x)

        return x, mu, logvar, None
