import os

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from .sentence_encoder import SentenceEncoder
from .condition_encoder import ConditionEncoder
from .image_encoder import ImageEncoder
from .dialog_encoder import DialogEncoder
from .generator import GeneratorRecurrentGANRes
from .discriminator import DiscriminatorAdditiveGANRes
from .loss import HingeAdversarial, gradient_penalty, kl_penalty
from .utils import get_grad_norm


class GeNeVATrainer:
    def __init__(
        self,
        # SentenceEncoder
        embedding_dim=1024,
        # ConditionEncoder
        hidden_dim=1024,
        # ImageEncoder
        image_feat_dim=512,
        # GeneratorRecurrentGANRes
        conditional=True,
        conditioning_dim=256,
        noise_dim=100,
        generator_sn=False,
        activation="leaky_relu",
        gen_fusion="concat",
        self_attention=False,
        cond_kl_reg=1,
        # DiscriminatorAdditiveGANRes
        disc_sn=True,
        disc_img_conditioning="subtract",
        conditioning="projection",
        disc_cond_channels=512,
        num_objects=24,
        # Optimizers
        gru_lr=0.003,
        feature_encoder_lr=0.006,
        rnn_lr=0.0003,
        generator_lr=0.0001,
        generator_beta1=0.0,
        generator_beta2=0.9,
        generator_weight_decay=0.0,
        discriminator_lr=0.0004,
        discriminator_beta1=0.0,
        discriminator_beta2=0.9,
        discriminator_weight_decay=0.0,
        # hyper-parameters
        gp_reg=1,
        aux_reg=5,
        wrong_fake_ratio=0.5,
        grad_clip=50,
    ):
        super().__init__()
        # hyper-parameters
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.cond_kl_reg = cond_kl_reg
        self.num_objects = num_objects
        self.gp_reg = gp_reg
        self.aux_reg = aux_reg
        self.wrong_fake_ratio = wrong_fake_ratio
        self.grad_clip = grad_clip

        # modules
        self.sentence_encoder = nn.DataParallel(SentenceEncoder(
            embedding_dim=embedding_dim)).cuda()
        self.condition_encoder = nn.DataParallel(ConditionEncoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim)).cuda()
        self.image_encoder = nn.DataParallel(
            ImageEncoder(image_feat_dim=image_feat_dim)).cuda()

        self.rnn = nn.DataParallel(DialogEncoder(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim), dim=1).cuda()

        self.generator = nn.DataParallel(GeneratorRecurrentGANRes(
            conditional=conditional,
            condition_dim=hidden_dim,
            conditioning_dim=conditioning_dim,
            noise_dim=noise_dim,
            generator_sn=generator_sn,
            activation=activation,
            gen_fusion=gen_fusion,
            self_attention=self_attention,
            cond_kl_reg=cond_kl_reg)).cuda()
        self.discriminator = nn.DataParallel(DiscriminatorAdditiveGANRes(
            activation=activation,
            disc_sn=disc_sn,
            disc_img_conditioning=disc_img_conditioning,
            conditioning=conditioning,
            disc_cond_channels=disc_cond_channels,
            self_attention=self_attention,
            condition_dim=hidden_dim,
            num_objects=num_objects)).cuda()

        # optimizers
        self.sentence_encoder_optimizer = Adam(
            self.sentence_encoder.parameters(), lr=gru_lr)
        feature_encoding_params = list(self.condition_encoder.parameters())
        feature_encoding_params += list(self.image_encoder.parameters())
        self.feature_encoders_optimizer = Adam(
            feature_encoding_params, lr=feature_encoder_lr)

        self.rnn_optimizer = Adam(
            self.rnn.parameters(), rnn_lr)

        self.generator_optimizer = Adam(
            self.generator.parameters(),
            lr=generator_lr,
            betas=(generator_beta1, generator_beta2),
            weight_decay=generator_weight_decay)
        self.discriminator_optimizer = Adam(
            self.discriminator.parameters(),
            lr=discriminator_lr,
            betas=(discriminator_beta1, discriminator_beta2),
            weight_decay=discriminator_weight_decay)

        # criterion
        self.criterion = HingeAdversarial()
        self.aux_criterion = nn.DataParallel(nn.BCELoss()).cuda()

    def predict_batch(self, batch, save_path):
        # NOTE: eval mode of GAN is sometimes disabled as a technique.
        # self.sentence_encoder.eval()
        # self.condition_encoder.eval()
        # self.image_encoder.eval()
        # self.rnn.eval()
        # self.generator.eval()

        batch_size = batch["turns_image"].size(0)
        max_seq_len = batch["turns_image"].size(1)

        prev_image = torch.FloatTensor(batch["background"])
        prev_image = prev_image.unsqueeze(0)
        prev_image = prev_image.repeat(batch_size, 1, 1, 1)
        hidden = torch.zeros(1, batch_size, self.hidden_dim)

        gen_images = []
        gt_images = []

        for t in range(max_seq_len):
            word_embeddings = batch["turns_word_embeddings"][:, t]
            text_length = batch["turns_text_length"][:, t]

            with torch.no_grad():
                image_feature_map, _ = self.image_encoder(prev_image)
                dt = self.sentence_encoder(word_embeddings, text_length)
                dt = self.condition_encoder(dt)

                dt = dt.unsqueeze(0)
                ht, hidden = self.rnn(dt, hidden)
                ht = ht.squeeze(0)

                fake_image, _, _, _ = self._generate(
                    image_feature_map, ht)

                # BUG: of original implementation.
                # BUG: it causes that some padding images will be saved as GT images.
                # if t == max_seq_len - 1:
                #     gen_images.append(fake_image)
                #     gt_images.append(batch["turns_image"][:, t])

                gen_images.append(fake_image)
                gt_images.append(batch["turns_image"][:, t])

                prev_image = fake_image

        self._save_predictions(
            gen_images,
            gt_images,
            batch["dialogs"],
            batch["scene_id"],
            save_path,
        )

        # self.sentence_encoder.train()
        # self.condition_encoder.train()
        # self.image_encoder.train()
        # self.rnn.train()
        # self.generator.train()

    def _save_predictions(self, gen_images, gt_images, dialogs, scene_ids, save_path):
        # i = index in batch
        for i, scene in enumerate(scene_ids):
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

                gen = (gen_images[t][i].data.cpu().numpy() + 1) * 128
                gen = np.clip(gen.astype(np.uint8), 0, 255)
                gen = gen.transpose(1, 2, 0)[..., ::-1]

                gt = (gt_images[t][i].data.cpu().numpy() + 1) * 128
                gt = np.clip(gt.astype(np.uint8), 0, 255)
                gt = gt.transpose(1, 2, 0)[..., ::-1]

                cv2.imwrite(
                    os.path.join(save_gen_subpath, '{}.png'.format(t)), gen)
                cv2.imwrite(
                    os.path.join(save_gt_subpath, '{}.png'.format(t)), gt)

    def train_batch(self, batch):
        """
        The training scheme follows the following:
            - Discriminator and Generator is updated every time step.
            - RNN, SentenceEncoder and ImageEncoder parameters are updated every sequence.
        """
        batch_size = batch["turns_image"].size(0)
        max_seq_len = batch["turns_image"].size(1)

        prev_image = torch.FloatTensor(batch["background"])
        prev_image = prev_image.unsqueeze(0)
        prev_image = prev_image.repeat(batch_size, 1, 1, 1)
        prev_image += torch.rand_like(prev_image) / 64

        # Initial inputs for the RNN set to zeros
        hidden = torch.zeros(1, batch_size, self.hidden_dim)
        # Initial objects for auxiliary classifier set to zeros (no objects)
        prev_objects = torch.zeros(batch_size, self.num_objects)

        teller_images = []
        drawer_images = []

        for t in range(max_seq_len):
            image = batch["turns_image"][:, t]
            word_embeddings = batch["turns_word_embeddings"][:, t]
            text_length = batch["turns_text_length"][:, t]
            objects = batch["turns_objects"][:, t]
            dialog_ended = t > (batch["dialog_length"] - 1)

            image += torch.rand_like(image) / 64

            image_feature_map, _ = self.image_encoder(prev_image)
            dt = self.sentence_encoder(word_embeddings, text_length)
            dt = self.condition_encoder(dt)

            dt = dt.unsqueeze(0)
            ht, hidden = self.rnn(dt, hidden)
            ht = ht.squeeze(0)

            fake_image, mu, logvar, sigma = self._generate(
                image_feature_map, ht.detach())

            hamming = objects - prev_objects
            hamming = torch.clamp(hamming, min=0)

            # backward & step discriminator
            d_loss, d_real, d_fake, aux_loss, discriminator_gradient = \
                self._optimize_discriminator(image,
                                             fake_image.detach(),
                                             prev_image,
                                             ht,
                                             dialog_ended,
                                             hamming)

            # backward & step generator
            g_loss, generator_gradient = \
                self._optimize_generator(fake_image,
                                         prev_image,
                                         ht.detach(),
                                         dialog_ended,
                                         objects,
                                         mu,
                                         logvar)

            # update prev data for next step
            prev_image = image
            prev_objects = objects

            # logging
            teller_images.append(image[0].detach().cpu().numpy())
            drawer_images.append(fake_image[0].detach().cpu().numpy())

        # backward & step (sentence_encoder, condition_encoder, rnn, image_encoder)
        rnn_gradient, sentence_gradient, condition_gradient, image_gradient = \
            self._optimize_rnn()

        # make outputs
        outputs = {
            "scaler": {
                "d_loss": d_loss,
                "g_loss": g_loss,
                "aux_loss": aux_loss,
                "rnn_gradient": rnn_gradient,
                "sentence_gradient": sentence_gradient,
                "condition_gradient": condition_gradient,
                "image_gradient": image_gradient,
                "discriminator_gradient": discriminator_gradient,
                "generator_gradient": generator_gradient,
            },
            "image": {
                "teller_image": np.stack(teller_images, axis=0),
                "drawer_image": np.stack(drawer_images, axis=0),
            },
            "dialog": batch["dialogs"][0],
        }

        return outputs

    def _generate(self, image_feat_map, ht):
        batch_size = image_feat_map.size(0)
        device = image_feat_map.device

        zt = torch.randn(
            (batch_size, self.noise_dim),
            dtype=torch.float32,
            device=device,
        )

        fake_image, mu, logvar, sigma = self.generator(
            zt, ht, image_feat_map)

        return fake_image, mu, logvar, sigma

    def _optimize_discriminator(self, real_image, fake_image, prev_image, ht, dialog_ended, hamming):
        # make wrong image-text pair
        # slide image data to left (0, 1, ..., N-1) -> (1, .., N-1, 0)
        wrong_image = torch.cat((real_image[1:], real_image[0:1]), dim=0)
        wrong_prev_image = torch.cat((prev_image[1:], prev_image[0:1]), dim=0)

        self.discriminator.zero_grad()
        real_image.requires_grad_()

        d_real, aux_real, _ = self.discriminator(real_image, ht, prev_image)
        d_fake, aux_fake, _ = self.discriminator(fake_image, ht, prev_image)
        d_wrong, _, _ = self.discriminator(wrong_image, ht, wrong_prev_image)

        d_loss, aux_loss = self._discriminator_loss(d_real,
                                                    d_fake,
                                                    d_wrong,
                                                    aux_real,
                                                    aux_fake,
                                                    hamming,
                                                    dialog_ended)
        d_loss.backward(retain_graph=True)
        if self.gp_reg > 0:
            reg = self.gp_reg * self._masked_gradient_penalty(d_real,
                                                              real_image,
                                                              dialog_ended)
            reg.backward(retain_graph=True)

        grad_norm = get_grad_norm(self.discriminator.parameters())
        self.discriminator_optimizer.step()

        d_loss = d_loss.item()
        d_real = d_real.detach().cpu().numpy()
        d_fake = d_fake.detach().cpu().numpy()
        aux_loss = aux_loss.item() if self.aux_reg > 0 else aux_loss
        grad_norm = grad_norm.item()

        return d_loss, d_real, d_fake, aux_loss, grad_norm

    def _discriminator_loss(self, d_real, d_fake, d_wrong, aux_real, aux_fake, hamming, dialog_ended):
        """
        Accumulates losses only for sequences that have not ended
        to avoid back-propagation through padding.
        """
        d_losses = []
        aux_losses = []

        for b, ended in enumerate(dialog_ended):
            if ended:
                continue

            d_loss = self.criterion.discriminator(
                d_real[b], d_fake[b], d_wrong[b], self.wrong_fake_ratio)

            if self.aux_reg > 0:
                aux_loss_real = self.aux_criterion(
                    aux_real[b], hamming[b]).mean()
                aux_loss_fake = self.aux_criterion(
                    aux_fake[b], hamming[b]).mean()
                aux_loss = self.aux_reg * (aux_loss_real + aux_loss_fake)
                d_loss += self.aux_reg * aux_loss

                aux_losses.append(aux_loss)

            d_losses.append(d_loss)

        d_loss = torch.stack(d_losses).mean()
        if len(aux_losses) > 0:
            aux_loss = torch.stack(aux_losses).mean()
        else:
            aux_loss = 0

        return d_loss, aux_loss

    def _masked_gradient_penalty(self, d_real, real_image, dialog_ended):
        # NOTE: actually not masked...
        gp_reg = gradient_penalty(d_real, real_image).mean()
        return gp_reg

    def _optimize_generator(self, fake_image, prev_image, ht, dialog_ended, objects, mu, logvar):
        self.generator.zero_grad()

        d_fake, aux_fake, _ = self.discriminator(fake_image, ht, prev_image)
        g_loss = self._generator_loss(
            d_fake, aux_fake, objects, dialog_ended, mu, logvar)

        g_loss.backward(retain_graph=True)
        grad_norm = get_grad_norm(self.generator.parameters())

        self.generator_optimizer.step()

        g_loss = g_loss.item()
        grad_norm = grad_norm.item()

        return g_loss, grad_norm

    def _generator_loss(self, d_fake, aux_fake, objects, dialog_ended, mu, logvar):
        """
        Accumulates losses only for sequences that have not ended
        to avoid back-propagation through padding.
        """
        g_losses = []

        for b, ended in enumerate(dialog_ended):
            if ended:
                continue

            g_loss = self.criterion.generator(d_fake[b])

            if self.aux_reg > 0:
                g_loss += self.aux_reg * \
                    self.aux_criterion(aux_fake[b], objects[b]).mean()

            if self.cond_kl_reg > 0:
                g_loss += self.cond_kl_reg * kl_penalty(mu[b], logvar[b])

            g_losses.append(g_loss)

        g_loss = torch.stack(g_losses).mean()

        return g_loss

    def _optimize_rnn(self):
        # step rnn with gradient clip
        clip_grad_norm_(self.rnn.parameters(), self.grad_clip)
        rnn_grad_norm = get_grad_norm(self.rnn.parameters()).item()
        self.rnn_optimizer.step()
        self.rnn.zero_grad()

        # step sentence_encoder with gradient clip
        clip_grad_norm_(self.sentence_encoder.parameters(), self.grad_clip)
        se_grad_norm = get_grad_norm(
            self.sentence_encoder.parameters()).item()
        self.sentence_encoder_optimizer.step()
        self.sentence_encoder.zero_grad()

        # step condition_encoder and image_encoder
        ce_grad_norm = get_grad_norm(
            self.condition_encoder.parameters()).item()
        ie_grad_norm = get_grad_norm(
            self.image_encoder.parameters()).item()
        self.feature_encoders_optimizer.step()
        self.condition_encoder.zero_grad()
        self.image_encoder.zero_grad()

        return rnn_grad_norm, se_grad_norm, ce_grad_norm, ie_grad_norm

    def get_snapshot(self, train_resumable=True):
        """Get model snapshots as state_dict.

        Parameters
        ----------
        train_resumable : bool, optional
            If True, state_dict contains following data to resume training, by default True.
            - discriminator
            - sentence_encoder_optimizer
            - feature_encoders_optimizer
            - rnn_optimizer
            - generator_optimizer
            - discriminator_optimizer

        Returns
        -------
        snapshot : Dict[str, Dict]
            state_dict
        """
        snapshot = {
            "sentence_encoder": self.sentence_encoder.state_dict(),
            "condition_encoder": self.condition_encoder.state_dict(),
            "image_encoder": self.image_encoder.state_dict(),
            "rnn": self.rnn.state_dict(),
            "generator": self.generator.state_dict(),
        }
        if train_resumable:
            snapshot.update({
                "discriminator": self.discriminator.state_dict(),
                "sentence_encoder_optimizer": self.sentence_encoder_optimizer.state_dict(),
                "feature_encoders_optimizer": self.feature_encoders_optimizer.state_dict(),
                "rnn_optimizer": self.rnn_optimizer.state_dict(),
                "generator_optimizer": self.generator_optimizer.state_dict(),
                "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            })
        return snapshot
