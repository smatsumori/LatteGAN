import os
import uuid
import shutil

import torch
from torch.utils.data import DataLoader

from modules.geneva.trainer import GeNeVATrainer
from modules.metrics.inception_localizer import calculate_inception_objects_accuracy
from data.codraw_dataset import CoDrawDataset, codraw_collate_fn
from data.iclevr_dataset import ICLEVRDataset, iclevr_collate_fn
# from utils.make_grid import make_grid_from_numpy

# import wandb

from logging import getLogger
logger = getLogger(__name__)


SAVE_ROOT_DIR = "./results/experiments/"
SAVE_DIR = None


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        # models
        self.model = GeNeVATrainer(
            # Modules
            # SentenceEncoder
            embedding_dim=cfg.embedding_dim,
            # ConditionEncoder
            hidden_dim=cfg.hidden_dim,
            # ImageEncoder
            image_feat_dim=cfg.image_feat_dim,
            # GeneratorRecurrentGANRes
            conditional=cfg.conditional,
            conditioning_dim=cfg.conditioning_dim,
            noise_dim=cfg.noise_dim,
            generator_sn=cfg.generator_sn,
            activation=cfg.activation,
            gen_fusion=cfg.gen_fusion,
            self_attention=cfg.self_attention,
            cond_kl_reg=cfg.cond_kl_reg,
            # DiscriminatorAdditiveGANRes
            disc_sn=cfg.disc_sn,
            disc_img_conditioning=cfg.disc_img_conditioning,
            conditioning=cfg.conditioning,
            disc_cond_channels=cfg.disc_cond_channels,
            num_objects=cfg.num_objects,
            # Optimizers
            gru_lr=cfg.gru_lr,
            feature_encoder_lr=cfg.feature_encoder_lr,
            rnn_lr=cfg.rnn_lr,
            generator_lr=cfg.generator_lr,
            generator_beta1=cfg.generator_beta1,
            generator_beta2=cfg.generator_beta2,
            generator_weight_decay=cfg.generator_weight_decay,
            discriminator_lr=cfg.discriminator_lr,
            discriminator_beta1=cfg.discriminator_beta1,
            discriminator_beta2=cfg.discriminator_beta2,
            discriminator_weight_decay=cfg.discriminator_weight_decay,
            # hyper-parameters
            gp_reg=cfg.gp_reg,
            aux_reg=cfg.aux_reg,
            wrong_fake_ratio=cfg.wrong_fake_ratio,
            grad_clip=cfg.grad_clip,
        )

        # dataset
        if cfg.dataset == "codraw":
            # codraw-train
            self.dataset = CoDrawDataset(
                dataset_path=cfg.dataset_path,
                batch_size=cfg.batch_size,
                glove_path=cfg.glove_path,
            )
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            self.dataloader.collate_fn = codraw_collate_fn
            # codraw-valid
            self.valid_dataset = CoDrawDataset(
                dataset_path=cfg.valid_dataset_path,
                batch_size=cfg.batch_size,
                glove_path=cfg.glove_path,
            )
            self.valid_dataloader = DataLoader(
                self.valid_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,  # evaluate all data
            )
            self.valid_dataloader.collate_fn = codraw_collate_fn
            # codraw-test
            self.test_dataset = CoDrawDataset(
                dataset_path=cfg.test_dataset_path,
                batch_size=cfg.batch_size,
                glove_path=cfg.glove_path,
            )
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,  # evaluate all data
            )
            self.test_dataloader.collate_fn = codraw_collate_fn
        elif cfg.dataset == "iclevr":
            # iclevr-train
            self.dataset = ICLEVRDataset(
                dataset_path=cfg.dataset_path,
                glove_path=cfg.glove_path,
            )
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            self.dataloader.collate_fn = iclevr_collate_fn
            # iclevr-valid
            self.valid_dataset = ICLEVRDataset(
                dataset_path=cfg.valid_dataset_path,
                glove_path=cfg.glove_path,
            )
            self.valid_dataloader = DataLoader(
                self.valid_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
            self.valid_dataloader.collate_fn = iclevr_collate_fn
            # iclevr-test
            self.test_dataset = ICLEVRDataset(
                dataset_path=cfg.test_dataset_path,
                glove_path=cfg.glove_path,
            )
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
            self.test_dataloader.collate_fn = iclevr_collate_fn
        else:
            raise NotImplementedError

    def evaluation(self, split="valid"):
        if split == "valid":
            eval_dataloader = self.valid_dataloader
        elif split == "test":
            eval_dataloader = self.test_dataloader
        else:
            raise ValueError

        # create directory for latest (not best!) images
        dirname = os.path.join(SAVE_DIR, f"images_{split}/")
        os.makedirs(dirname, exist_ok=True)

        # generate images
        for batch in eval_dataloader:
            self.model.predict_batch(batch, dirname)

        # submission
        ap, ar, f1, _, rsim = \
            calculate_inception_objects_accuracy(
                dirname,
                self.cfg.detector_localizer_path,
            )

        outputs = {
            f"{split}_AP": ap,
            f"{split}_AR": ar,
            f"{split}_F1": f1,
            f"{split}_RSIM": rsim,
        }

        return outputs

    def train(self):
        cnt_iter = 0

        for epoch in range(self.cfg.epochs):
            if self.cfg.dataset == "codraw":
                self.dataset.shuffle()

            for batch in self.dataloader:
                cnt_iter += 1
                outputs = self.model.train_batch(batch)

                # evaluation
                if cnt_iter % self.cfg.eval_step == 0:
                    logger.debug("evaluate by valid data...")
                    outputs["scaler"].update(self.evaluation(split="valid"))
                    logger.debug("evaluate by test data...")
                    outputs["scaler"].update(self.evaluation(split="test"))

                # wandb upload
                if cnt_iter % self.cfg.vis_step == 0:
                    logger.debug(
                        f"epoch: {epoch}, step: {cnt_iter}, data: {outputs['scaler']}")
                    # self._upload_wandb(
                    #     outputs,
                    #     step=cnt_iter,
                    #     with_sample=(cnt_iter % self.cfg.vis_sample_step == 0),
                    # )

                # save model
                if cnt_iter % self.cfg.save_step == 0:
                    saved_path = self._save_model(epoch, cnt_iter)
                    logger.info(
                        f"[e{epoch}, i{cnt_iter}] snapshot {saved_path} saved.")

    def _save_model(self, epoch, cnt_iter):
        # create directory for snapshots
        save_snapshot_dir = os.path.join(SAVE_DIR, "snapshots")
        os.makedirs(save_snapshot_dir, exist_ok=True)

        # save current snapshot
        snapshot = self.model.get_snapshot(train_resumable=False)
        path = os.path.join(save_snapshot_dir, f"e{epoch}_i{cnt_iter}.pth")
        torch.save(snapshot, path)

        return path

    # def _upload_wandb(self, outputs, step, with_sample=False):
    #     log = {}

    #     # scalers, including AP, AR, F1, RSIM
    #     log.update(outputs["scaler"])

    #     if with_sample:
    #         # sample image of train
    #         log.update({
    #             k: wandb.Image(
    #                 make_grid_from_numpy(
    #                     v,
    #                     normalize=lambda x: (x + 1.) / 2.,
    #                 )
    #             )
    #             for k, v in outputs["image"].items()
    #         })

    #         # sample dialog of train
    #         log.update({
    #             "dialog": wandb.Table(
    #                 columns=["dialog"],
    #                 data=[[t] for t in outputs["dialog"]],
    #             )
    #         })

    #     # upload
    #     wandb.log(log, step=step)


def train_geneva(cfg):
    logger.info(f"script {__name__} start!")

    # SAVE_DIR SETTING
    global SAVE_DIR
    SAVE_DIR = os.path.join(SAVE_ROOT_DIR, cfg.name)
    if os.path.exists(SAVE_DIR):
        logger.warning(f"{SAVE_DIR} already exists. Overwrite by current run?")
        stdin = input("Press [Y/n]: ")
        # if Yes --> delete old files
        if stdin == "Y":
            shutil.rmtree(SAVE_DIR)
        # if No --> create temporary directory
        elif stdin == "n":
            SAVE_DIR = os.path.join(f"/var/tmp/{uuid.uuid4()}", cfg.name)
            logger.warning(f"temporary save at {SAVE_DIR}.")
        else:
            raise ValueError
    os.makedirs(SAVE_DIR)

    # WANDB SETTING
    # wandb.init(
    #     name=cfg.name,
    #     notes=cfg.notes,
    #     project=cfg.project,
    #     entity=cfg.entity,
    #     config=cfg,
    # )
    # os.symlink(
    #     os.path.abspath(wandb.run.dir),
    #     os.path.abspath(os.path.join(SAVE_DIR, "wandb")),
    # )

    # running
    trainer = Trainer(cfg)
    trainer.train()
