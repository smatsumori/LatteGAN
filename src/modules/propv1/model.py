import os

import numpy as np
import cv2

import torch
import torch.nn as nn

from .image_encoder import ImageEncoder, ImageEncoderMultiScale
from .generator import Generator, GeneratorMultiScale
from .discriminator import Discriminator
from .unet_discriminator import UnetDiscriminator

from modules.geneva.loss import HingeAdversarial, gradient_penalty, kl_penalty
from modules.geneva.utils import get_grad_norm
from utils.truncnorm import truncated_normal


class GeNeVAPropV1Model:
    def __init__(
        self,
        is_multiscale=False,
        depth=4,  # multi-scale params
        fusion_image_size=8,  # multi-scale params
        output_image_size=128,  # multi-scale params
        image_feat_dim=512,
        generator_sn=True,
        generator_norm="bn",
        embedding_dim=768,
        condaug_out_dim=256,
        cond_kl_reg=1.0,
        noise_dim=100,
        fusion="concat",
        sta_concat=False,
        nhead=1,
        res_mask_post=False,
        multi_channel_gate=False,
        discriminator_arch="standard",
        discriminator_sn=True,
        num_objects=58,
        generator_lr=1e-4,
        generator_beta1=0.0,
        generator_beta2=0.9,
        discriminator_lr=4e-4,
        discriminator_beta1=0.0,
        discriminator_beta2=0.9,
        wrong_fake_ratio=0.5,
        aux_reg=10.0,
        gp_reg=10.0,
    ):

        # modules
        self.noise_dim = noise_dim
        self.is_multiscale = is_multiscale
        if self.is_multiscale:
            self.image_encoder = nn.DataParallel(
                ImageEncoderMultiScale(
                    depth=depth,
                    image_feat_dim=image_feat_dim,
                    norm=generator_norm,
                    use_spectral_norm=generator_sn,
                )
            ).cuda()
            self.generator = nn.DataParallel(
                GeneratorMultiScale(
                    condition_dim=embedding_dim,
                    condaug_out_dim=condaug_out_dim,
                    cond_kl_reg=cond_kl_reg,
                    noise_dim=noise_dim,
                    norm=generator_norm,
                    generator_sn=generator_sn,
                    fusion=fusion,
                    fusion_image_size=fusion_image_size,
                    output_image_size=output_image_size,
                    image_feat_dim=image_feat_dim,
                    sta_concat=sta_concat,
                    nhead=nhead,
                    res_mask_post=res_mask_post,
                    multi_channel_gate=multi_channel_gate,
                )
            ).cuda()
            self.eval_generator = nn.DataParallel(
                GeneratorMultiScale(
                    condition_dim=embedding_dim,
                    condaug_out_dim=condaug_out_dim,
                    cond_kl_reg=cond_kl_reg,
                    noise_dim=noise_dim,
                    norm=generator_norm,
                    generator_sn=generator_sn,
                    fusion=fusion,
                    fusion_image_size=fusion_image_size,
                    output_image_size=output_image_size,
                    image_feat_dim=image_feat_dim,
                    sta_concat=sta_concat,
                    nhead=nhead,
                    res_mask_post=res_mask_post,
                    multi_channel_gate=multi_channel_gate,
                )
            ).cuda()
            self.eval_generator.load_state_dict(self.generator.state_dict())
        else:
            self.image_encoder = nn.DataParallel(
                ImageEncoder(
                    image_feat_dim=image_feat_dim,
                    norm=generator_norm,
                    use_spectral_norm=generator_sn,
                )
            ).cuda()
            self.generator = nn.DataParallel(
                Generator(
                    condition_dim=embedding_dim,
                    condaug_out_dim=condaug_out_dim,
                    cond_kl_reg=cond_kl_reg,
                    noise_dim=noise_dim,
                    norm=generator_norm,
                    generator_sn=generator_sn,
                    fusion=fusion,
                    image_feat_dim=image_feat_dim,
                    sta_concat=sta_concat,
                    nhead=nhead,
                    res_mask_post=res_mask_post,
                    multi_channel_gate=multi_channel_gate,
                )
            ).cuda()
            self.eval_generator = nn.DataParallel(
                Generator(
                    condition_dim=embedding_dim,
                    condaug_out_dim=condaug_out_dim,
                    cond_kl_reg=cond_kl_reg,
                    noise_dim=noise_dim,
                    norm=generator_norm,
                    generator_sn=generator_sn,
                    fusion=fusion,
                    image_feat_dim=image_feat_dim,
                    sta_concat=sta_concat,
                    nhead=nhead,
                    res_mask_post=res_mask_post,
                    multi_channel_gate=multi_channel_gate,
                )
            ).cuda()
            self.eval_generator.load_state_dict(self.generator.state_dict())

        self.discriminator_arch = discriminator_arch
        if self.discriminator_arch == "standard":
            self.discriminator = nn.DataParallel(
                Discriminator(
                    condition_dim=embedding_dim,
                    discriminator_sn=discriminator_sn,
                    aux_detection_dim=num_objects,
                )
            ).cuda()
        elif self.discriminator_arch == "unet":
            self.discriminator = nn.DataParallel(
                UnetDiscriminator(
                    condition_dim=embedding_dim,
                    discriminator_sn=discriminator_sn,
                    aux_detection_dim=num_objects,
                )
            ).cuda()
        else:
            raise ValueError

        # optimizers
        parameters = list(self.image_encoder.parameters())
        parameters += list(self.generator.parameters())
        self.generator_optimizer = torch.optim.Adam(
            parameters,
            lr=generator_lr,
            betas=(generator_beta1, generator_beta2),
            weight_decay=0.0,
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=discriminator_lr,
            betas=(discriminator_beta1, discriminator_beta2),
            weight_decay=0.0,
        )

        # criterion
        self.wrong_fake_ratio = wrong_fake_ratio
        self.aux_reg = aux_reg
        self.gp_reg = gp_reg
        self.cond_kl_reg = cond_kl_reg
        self.criterion = HingeAdversarial()
        self.aux_criterion = nn.DataParallel(nn.BCEWithLogitsLoss()).cuda()

    def predict_batch(self, batch, save_path):
        # NOTE: eval mode of GAN is sometimes disabled as a technique.
        # eval mode made images collapsed, but score is good...
        # self.image_encoder.eval()
        # self.generator.eval()

        batch_size = batch["turns_image"].size(0)
        max_dialog_length = batch["turns_image"].size(1)

        prev_image = batch["background"]
        prev_image = prev_image.unsqueeze(0)
        prev_image = prev_image.repeat(batch_size, 1, 1, 1)

        gen_images = []
        gt_images = []

        for t in range(max_dialog_length):
            text_embedding = batch["turns_text_embedding"][:, t]
            word_embeddings = batch["turns_word_embeddings"][:, t]
            text_length = batch["turns_text_length"][:, t]

            with torch.no_grad():
                prev_image += torch.rand_like(prev_image) / 64.0

                img_feat_map = self.image_encoder(prev_image)
                # truncation trick when prediction
                z = truncated_normal(
                    (batch_size, self.noise_dim),
                    threshold=2.0,
                ).astype(np.float32)
                z = torch.from_numpy(z)
                fake_image, mu, logvar, _ = self.eval_generator(
                    z,
                    text_embedding,
                    img_feat_map,
                    word_embeddings,
                    text_length,
                )

                # (L, B, C, H, W)
                gen_images.append(fake_image)
                gt_images.append(batch["turns_image"][:, t])

                prev_image = fake_image

        self._save_predictions(
            gen_images,  # (L, B, C, H, W)
            gt_images,  # (L, B, C, H, W)
            batch["dialogs"],  # (B, l) no zero padding
            batch["scene_id"],  # (B)
            save_path,
        )

        # self.image_encoder.train()
        # self.generator.train()

    def _save_predictions(self, gen_images, gt_images, dialogs, scene_ids, save_path):
        # i = index in batch
        for i, scene in enumerate(scene_ids):
            # save_gen_subpath: images_{split}/{scene_id}/{turn_id}.png
            # save_gt_subpath: images_{split}/{scene_id}_gt/{turn_id}.png
            save_gen_subpath = os.path.join(save_path, str(scene))
            save_gt_subpath = os.path.join(save_path, str(scene) + "_gt")
            os.makedirs(save_gen_subpath, exist_ok=True)
            os.makedirs(save_gt_subpath, exist_ok=True)

            with open(os.path.join(save_path, str(scene) + ".txt"), "w") as f:
                for j, text in enumerate(dialogs[i]):
                    f.write(f"{j}: {text}\n")

            for t in range(len(gen_images)):
                # len(dialogs[i]) = dialog length of a sample
                if t >= len(dialogs[i]):
                    continue

                # gen_images.shape: (L, B, C, H, W)
                gen = (gen_images[t][i].data.cpu().numpy() + 1) * 128
                gen = np.clip(gen.astype(np.uint8), 0, 255)
                # gen.shape: (C, H, W) -> (H, W, C)
                gen = gen.transpose(1, 2, 0)[..., ::-1]  # convert to bgr

                gt = (gt_images[t][i].data.cpu().numpy() + 1) * 128
                gt = np.clip(gt.astype(np.uint8), 0, 255)
                gt = gt.transpose(1, 2, 0)[..., ::-1]

                cv2.imwrite(
                    os.path.join(save_gen_subpath, '{}.png'.format(t)), gen)
                cv2.imwrite(
                    os.path.join(save_gt_subpath, '{}.png'.format(t)), gt)

    def train_batch(self, batch):
        batch_size = batch["source_image"].size(0)

        source_image = batch["source_image"]
        target_image = batch["target_image"]
        text_embedding = batch["text_embedding"]
        word_embeddings = batch["word_embeddings"]
        text_length = batch["text_length"]
        added_objects = batch["added_objects"]

        # data augmentation (add slight noise)
        source_image += torch.rand_like(source_image) / 64.0
        target_image += torch.rand_like(target_image) / 64.0

        # generate predict target image
        src_img_feat_map = self.image_encoder(source_image)
        z = torch.randn(
            (batch_size, self.noise_dim),
            dtype=torch.float32,
        )
        fake_image, mu, logvar, _ = self.generator(
            z,
            text_embedding,
            src_img_feat_map,
            word_embeddings,
            text_length,
        )

        # discriminator & backward & step
        d_loss, aux_loss, d_grad = self._optimize_discriminator(
            target_image,
            fake_image.detach(),
            source_image,
            text_embedding,
            added_objects,
        )

        # generator through discriminator & backward & step
        g_loss, g_grad, ie_grad = self._optimize_generator(
            fake_image,
            source_image,
            text_embedding,
            added_objects,
            mu,
            logvar,
        )

        # update eval generator
        for param_gen, param_egen in zip(self.generator.parameters(), self.eval_generator.parameters()):
            param_egen.data.mul_(0.999).add_(0.001 * param_gen.data)
        for buffer_gen, buffer_egen in zip(self.generator.buffers(), self.eval_generator.buffers()):
            buffer_egen.data.mul_(0.999).add_(0.001 * buffer_gen.data)

        # make outputs
        outputs = {
            "scalar": {
                "d_loss": d_loss,
                "g_loss": g_loss,
                "aux_loss": aux_loss,
                "discriminator_gradient": d_grad,
                "generator_gradient": g_grad,
                "image_gradient": ie_grad,
            },
            "image": {
                "source_image": source_image[:4].detach().cpu().numpy(),
                "teller_image": target_image[:4].detach().cpu().numpy(),
                "drawer_image": fake_image[:4].detach().cpu().numpy(),
            },
            "dialog": batch["utter"][:4],
            "others": {}
        }

        return outputs

    def _optimize_discriminator(
        self,
        real_image,
        fake_image,
        prev_image,
        text_embedding,
        added_objects,
    ):
        # make wrong image-text pair
        # slide image data to left (0, 1, ..., N-1) -> (1, .., N-1, 0)
        wrong_image = torch.cat((real_image[1:], real_image[0:1]), dim=0)
        wrong_prev_image = torch.cat((prev_image[1:], prev_image[0:1]), dim=0)

        self.discriminator.zero_grad()
        real_image.requires_grad_()

        # feed discriminator
        if self.discriminator_arch == "standard":
            d_real, aux_real = self.discriminator(
                prev_image, real_image, text_embedding)
            d_fake, _ = self.discriminator(
                prev_image, fake_image, text_embedding)
            d_wrong, _ = self.discriminator(
                wrong_prev_image, wrong_image, text_embedding)
        elif self.discriminator_arch == "unet":
            d_real, du_real, aux_real = self.discriminator(
                prev_image, real_image, text_embedding)
            d_fake, du_fake, _ = self.discriminator(
                prev_image, fake_image, text_embedding)
            d_wrong, du_wrong, _ = self.discriminator(
                wrong_prev_image, wrong_image, text_embedding)

        # loss
        d_loss = self.criterion.discriminator(
            d_real,
            d_fake,
            d_wrong,
            self.wrong_fake_ratio,
        )
        if self.discriminator_arch == "unet":
            d_loss += self.criterion.discriminator(
                du_real,
                du_fake,
                wrong=None,  # unet-decoder for image-reality
                wrong_weight=None,
            )

        if self.aux_reg > 0.0:
            aux_loss = self.aux_criterion(aux_real, added_objects).mean()
            d_loss += self.aux_reg * aux_loss
        else:
            aux_loss = 0

        d_loss.backward(retain_graph=True)
        if self.gp_reg > 0.0:
            reg = self.gp_reg * gradient_penalty(d_real, real_image)
            reg.backward(retain_graph=True)

        d_grad = get_grad_norm(self.discriminator.parameters())
        self.discriminator_optimizer.step()

        d_loss = d_loss.item()
        aux_loss = aux_loss.item() if self.aux_reg > 0.0 else aux_loss
        d_grad = d_grad.item()

        return d_loss, aux_loss, d_grad

    def _optimize_generator(
        self,
        fake_image,
        prev_image,
        text_embedding,
        added_objects,
        mu,
        logvar,
    ):
        self.image_encoder.zero_grad()
        self.generator.zero_grad()

        # feed discriminator
        if self.discriminator_arch == "standard":
            d_fake, aux_fake = self.discriminator(
                prev_image, fake_image, text_embedding)
        elif self.discriminator_arch == "unet":
            d_fake, du_fake, aux_fake = self.discriminator(
                prev_image, fake_image, text_embedding)

        # loss
        g_loss = self.criterion.generator(d_fake)
        if self.discriminator_arch == "unet":
            g_loss += self.criterion.generator(du_fake)

        if self.aux_reg > 0.0:
            aux_loss = self.aux_criterion(aux_fake, added_objects).mean()
            g_loss += self.aux_reg * aux_loss

        if self.cond_kl_reg > 0.0:
            kl_loss = kl_penalty(mu, logvar)
            g_loss += self.cond_kl_reg * kl_loss

        g_loss.backward(retain_graph=True)

        g_grad = get_grad_norm(self.generator.parameters())
        ie_grad = get_grad_norm(self.image_encoder.parameters())
        self.generator_optimizer.step()

        g_loss = g_loss.item()
        g_grad = g_grad.item()
        ie_grad = ie_grad.item()

        return g_loss, g_grad, ie_grad
