import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ResBlockDown, ResBlockUp, conv1x1


class UnetDiscriminator(nn.Module):
    def __init__(
        self,
        condition_dim=768,
        discriminator_sn=True,
        aux_detection_dim=58,
    ):
        super().__init__()

        # encoder for both source and target image
        # x: (3, 128, 128) --> (64, 64, 64)
        self.resdown1 = ResBlockDown(
            3,
            64,
            use_spectral_norm=discriminator_sn,
            is_first_block=True,
        )
        # x: (64, 64, 64) --> (128, 32, 32)
        self.resdown2 = ResBlockDown(
            64,
            128,
            use_spectral_norm=discriminator_sn,
        )
        # x: (128, 32, 32) --> (256, 16, 16)
        self.resdown3 = ResBlockDown(
            128,
            256,
            use_spectral_norm=discriminator_sn,
        )

        # discriminator(global) from feature difference
        # dxt: (256, 16, 16) --> (512, 8, 8)
        self.resdown4 = ResBlockDown(
            256,
            512,
            use_spectral_norm=discriminator_sn,
        )
        # dxt: (512, 8, 8) --> (1024, 4, 4)
        self.resdown5 = ResBlockDown(
            512,
            1024,
            use_spectral_norm=discriminator_sn,
        )
        # dxt: (1024, 4, 4) --> (1024, 4, 4)
        self.resdown6 = ResBlockDown(
            1024,
            1024,
            use_spectral_norm=discriminator_sn,
            use_downsample=False,
        )

        # discriminator(local) from target feature
        # xt: (256, 16, 16) --> (512, 8, 8)
        self.resbottleneck = ResBlockDown(
            256,
            512,
            use_spectral_norm=discriminator_sn,
        )
        # xt: (512, 8, 8) --> (256, 16, 16)
        self.resup4 = ResBlockUp(
            512,
            256,
            norm="none",
            use_spectral_norm=discriminator_sn,
        )
        # xt*2: (256 + 256, 16, 16) --> (128, 32, 32)
        self.resup5 = ResBlockUp(
            256 + 256,
            128,
            norm="none",
            use_spectral_norm=discriminator_sn,
        )
        # xt*2: (128 + 128, 32, 32) --> (64, 64, 64)
        self.resup6 = ResBlockUp(
            128 + 128,
            64,
            norm="none",
            use_spectral_norm=discriminator_sn,
        )
        # xt*2: (64 + 64, 64, 64) --> (64, 128, 128)
        self.resup7 = ResBlockUp(
            64 + 64,
            64,
            norm="none",
            use_spectral_norm=discriminator_sn,
        )

        # adversarial
        self.linear_encoder = conv1x1(1024, 1, discriminator_sn)
        self.linear_decoder = conv1x1(64, 1, discriminator_sn)

        # projection
        self.projection_encoder = conv1x1(condition_dim, 1024)

        # auxiliary object detector
        self.aux_detector = nn.Sequential(
            conv1x1(1024, 256, discriminator_sn),
            nn.ReLU(),
            conv1x1(256, aux_detection_dim, discriminator_sn),
        )

    def forward(self, x_src, x_tgt, y):
        # encoder for both source and target image
        # source image
        xs = self.resdown1(x_src)
        xs = self.resdown2(xs)
        xs = self.resdown3(xs)
        # target image
        xt1 = self.resdown1(x_tgt)
        xt2 = self.resdown2(xt1)
        xt3 = self.resdown3(xt2)

        # discriminator(global) from feature difference
        dx = xt3 - xs
        dx = self.resdown4(dx)
        dx = self.resdown5(dx)
        dx = self.resdown6(dx)
        dx = F.relu(dx)
        dx = torch.sum(dx, dim=(2, 3), keepdim=True)

        # discriminator(local) from target feature
        xt = self.resbottleneck(xt3)
        xt = self.resup4(xt)
        xt = self.resup5(torch.cat([xt, xt3], dim=1))
        xt = self.resup6(torch.cat([xt, xt2], dim=1))
        xt = self.resup7(torch.cat([xt, xt1], dim=1))
        xt = F.relu(xt)

        # adversarial
        adv_enc = self.linear_encoder(dx)
        adv_dec = self.linear_decoder(xt)

        # projection
        y = y.unsqueeze(2).unsqueeze(3)
        y = self.projection_encoder(y)
        y = torch.sum(dx * y, dim=1, keepdim=True)

        adv_enc = (adv_enc + y).squeeze(3).squeeze(2)

        # auxiliary object detector
        aux = self.aux_detector(dx)
        aux = aux.squeeze(3).squeeze(2)

        return adv_enc, adv_dec, aux


# class UnetDiscriminator(nn.Module):
#     def __init__(
#         self,
#         condition_dim=768,
#         discriminator_sn=True,
#         aux_detection_dim=58,
#     ):
#         super().__init__()

#         # encoder for both source and target image
#         # x: (3, 128, 128) --> (64, 64, 64)
#         self.resdown1 = ResBlockDown(
#             3,
#             64,
#             use_spectral_norm=discriminator_sn,
#             is_first_block=True,
#         )
#         # x: (64, 64, 64) --> (128, 32, 32)
#         self.resdown2 = ResBlockDown(
#             64,
#             128,
#             use_spectral_norm=discriminator_sn,
#         )
#         # x: (128, 32, 32) --> (256, 16, 16)
#         self.resdown3 = ResBlockDown(
#             128,
#             256,
#             use_spectral_norm=discriminator_sn,
#         )

#         # discriminator(down) from feature difference
#         # dxt: (256, 16, 16) --> (512, 8, 8)
#         self.resdown4 = ResBlockDown(
#             256,
#             512,
#             use_spectral_norm=discriminator_sn,
#         )

#         # discriminator to global logit
#         # dxt: (512, 8, 8) --> (1024, 8, 8)
#         self.resdown5 = ResBlockDown(
#             512,
#             1024,
#             use_spectral_norm=discriminator_sn,
#             use_downsample=False,
#         )

#         # discriminator(up) from feature difference
#         # dxt: (512, 8, 8) --> (256, 16, 16)
#         self.resup4 = ResBlockUp(
#             512,
#             256,
#             norm="none",
#             use_spectral_norm=discriminator_sn,
#         )
#         # dxt, dxt: (256 + 256, 16, 16) --> (128, 32, 32)
#         self.resup5 = ResBlockUp(
#             256 + 256,
#             128,
#             norm="none",
#             use_spectral_norm=discriminator_sn,
#         )
#         # dxt, dxt: (128 + 128, 32, 32) --> (64, 64, 64)
#         self.resup6 = ResBlockUp(
#             128 + 128,
#             64,
#             norm="none",
#             use_spectral_norm=discriminator_sn,
#         )
#         # dxt, dxt: (64 + 64, 64, 64) --> (64, 128, 128)
#         self.resup7 = ResBlockUp(
#             64 + 64,
#             64,
#             norm="none",
#             use_spectral_norm=discriminator_sn,
#         )

#         # adversarial
#         self.linear_encoder = conv1x1(1024, 1, discriminator_sn)
#         self.linear_decoder = conv1x1(64, 1, discriminator_sn)

#         # projection
#         self.projection_encoder = conv1x1(condition_dim, 1024)
#         self.projection_decoder = conv1x1(condition_dim, 64)

#         # auxiliary object detector
#         self.aux_detector = nn.Sequential(
#             conv1x1(1024, 256, discriminator_sn),
#             nn.ReLU(),
#             conv1x1(256, aux_detection_dim, discriminator_sn),
#         )

#     def forward(self, x_src, x_tgt, y):
#         # encoder for both source and target image
#         # source image
#         xs1 = self.resdown1(x_src)
#         xs2 = self.resdown2(xs1)
#         xs3 = self.resdown3(xs2)
#         # target image
#         xt1 = self.resdown1(x_tgt)
#         xt2 = self.resdown2(xt1)
#         xt3 = self.resdown3(xt2)

#         # discriminator(down) from feature difference
#         # dx.shape = (B, 256, 16, 16)
#         dx = xt3 - xs3
#         dx = self.resdown4(dx)
#         dxd = self.resdown5(dx)
#         dxd = F.relu(dxd)
#         dxd = torch.sum(dxd, dim=(2, 3), keepdim=True)

#         # discriminator(up) from feature difference
#         # dx.shape = (B, 512, 8, 8)
#         dxu = self.resup4(dx)
#         dxu = self.resup5(torch.cat([dxu, xt3 - xs3], dim=1))
#         dxu = self.resup6(torch.cat([dxu, xt2 - xs2], dim=1))
#         dxu = self.resup7(torch.cat([dxu, xt1 - xs1], dim=1))
#         dxu = F.relu(dxu)

#         # adversarial
#         advd = self.linear_encoder(dxd)
#         advu = self.linear_decoder(dxu)

#         # projection(down)
#         y = y.unsqueeze(2).unsqueeze(3)
#         yd = self.projection_encoder(y)
#         yd = torch.sum(dxd * yd, dim=1, keepdim=True)
#         advd = (advd + yd).squeeze(3).squeeze(2)

#         # projection(up)
#         _, _, h, w = dxu.size()
#         yu = self.projection_decoder(y)
#         yu = yu.repeat(1, 1, h, w)
#         yu = torch.sum(dxu * yu, dim=1, keepdim=True)
#         advu = advu + yu

#         # auxiliary object detector
#         aux = self.aux_detector(dxd)
#         aux = aux.squeeze(3).squeeze(2)

#         return advd, advu, aux
