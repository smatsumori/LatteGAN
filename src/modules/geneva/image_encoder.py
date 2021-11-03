import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, image_feat_dim=512):
        """
        Parameters
        ----------
        image_feat_dim : int, optional
            Channel size of output image feature f_{G_t}, by default 512
        """
        super().__init__()
        self.image_encoder = nn.Sequential(
            # 3 x 128 x 128
            nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            # 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            # 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.Conv2d(128, image_feat_dim, 4, stride=2, padding=1, bias=False),
            # image_feat_dim x 16 x 16
            nn.BatchNorm2d(image_feat_dim),
        )

    def forward(self, x):
        """Return image feature maps and pooled one.

        Parameters
        ----------
        x : torch.tensor
            shape (B, C, H, W).

        Returns
        -------
        image_features : torch.tensor
            shape (B, C', H', W').
            C' = image_feat_dim (default 512), H' and W' = H // 8 and W // 8.
            In GeNeVa paper, it's represented with "f_{G_t}".
        pooled_features : torch.tensor
            shape (B, C').
            C' = image_feat_dim (default 512).
            Image feature of sum pooled image_features.
        """
        image_features = self.image_encoder(x)
        pooled_features = torch.sum(image_features, dim=(2, 3))
        # NOTE: object_detections seems not to be used in GeNeVA?
        return image_features, pooled_features
