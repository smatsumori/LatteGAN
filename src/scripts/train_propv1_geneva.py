import os
import uuid
import shutil

from torch.utils.data import DataLoader

from modules.propv1.model import GeNeVAPropV1Model
from modules.metrics.inception_localizer import calculate_inception_objects_accuracy
from data.codraw_propv1_dataset import CoDrawPropV1TrainDataset, codraw_propv1_train_collate_fn
from data.codraw_propv1_dataset import CoDrawPropV1EvalDataset, codraw_propv1_eval_collate_fn
from data.iclevr_propv1_dataset import ICLEVRPropV1TrainDataset, iclevr_propv1_train_collate_fn
from data.iclevr_propv1_dataset import ICLEVRPropV1EvalDataset, iclevr_propv1_eval_collate_fn
# from utils.plotter import plot_multilabel_confusion_matrix
# from utils.make_grid import make_grid_from_numpy
# from utils.consts import CODRAW_OBJS

# import wandb

from logging import getLogger
logger = getLogger(__name__)


SAVE_ROOT_DIR = "./results/experiments/"
SAVE_DIR = None


class PropV1Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        # result path
        self.save_path = os.path.join(SAVE_ROOT_DIR, cfg.name)

        if "is_multiscale" not in cfg:
            cfg.is_multiscale = False
            cfg.depth = 4
            cfg.fusion_image_size = 8
        if "sta_concat" not in cfg:
            cfg.sta_concat = False
        if "nhead" not in cfg:
            cfg.nhead = 1
        if "res_mask_post" not in cfg:
            cfg.res_mask_post = False
        if "multi_channel_gate" not in cfg:
            cfg.multi_channel_gate = False
        if "discriminator_arch" not in cfg:
            cfg.discriminator_arch = "standard"

        # models
        self.model = GeNeVAPropV1Model(
            is_multiscale=cfg.is_multiscale,
            depth=cfg.depth,
            fusion_image_size=cfg.fusion_image_size,
            output_image_size=cfg.image_size,
            image_feat_dim=cfg.image_feat_dim,
            generator_sn=cfg.generator_sn,
            generator_norm=cfg.generator_norm,
            embedding_dim=cfg.embedding_dim,
            condaug_out_dim=cfg.condaug_out_dim,
            cond_kl_reg=cfg.cond_kl_reg,
            noise_dim=cfg.noise_dim,
            fusion=cfg.fusion,
            sta_concat=cfg.sta_concat,
            nhead=cfg.nhead,
            res_mask_post=cfg.res_mask_post,
            multi_channel_gate=cfg.multi_channel_gate,
            discriminator_arch=cfg.discriminator_arch,
            discriminator_sn=cfg.discriminator_sn,
            num_objects=cfg.num_objects,
            generator_lr=cfg.generator_lr,
            generator_beta1=cfg.generator_beta1,
            generator_beta2=cfg.generator_beta2,
            discriminator_lr=cfg.discriminator_lr,
            discriminator_beta1=cfg.discriminator_beta1,
            discriminator_beta2=cfg.discriminator_beta2,
            wrong_fake_ratio=cfg.wrong_fake_ratio,
            aux_reg=cfg.aux_reg,
            gp_reg=cfg.gp_reg,
        )

        # dataset
        if cfg.dataset == "codraw":
            # train
            self.dataset = CoDrawPropV1TrainDataset(
                dataset_path=cfg.dataset_path,
                embed_dataset_path=cfg.embed_dataset_path,
                image_size=cfg.image_size,
            )
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            self.dataloader.collate_fn = codraw_propv1_train_collate_fn
            # valid
            self.valid_dataset = CoDrawPropV1EvalDataset(
                dataset_path=cfg.valid_dataset_path,
                embed_dataset_path=cfg.valid_embed_dataset_path,
                batch_size=cfg.batch_size,
                image_size=cfg.image_size,
            )
            self.valid_dataloader = DataLoader(
                self.valid_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,  # evaluate all data
            )
            self.valid_dataloader.collate_fn = codraw_propv1_eval_collate_fn
            # test
            self.test_dataset = CoDrawPropV1EvalDataset(
                dataset_path=cfg.test_dataset_path,
                embed_dataset_path=cfg.test_embed_dataset_path,
                batch_size=cfg.batch_size,
                image_size=cfg.image_size,
            )
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,  # evaluate all data
            )
            self.test_dataloader.collate_fn = codraw_propv1_eval_collate_fn
        elif cfg.dataset == "iclevr":
            # train
            self.dataset = ICLEVRPropV1TrainDataset(
                dataset_path=cfg.dataset_path,
                embed_dataset_path=cfg.embed_dataset_path,
                image_size=cfg.image_size,
            )
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            self.dataloader.collate_fn = iclevr_propv1_train_collate_fn
            # valid
            self.valid_dataset = ICLEVRPropV1EvalDataset(
                dataset_path=cfg.valid_dataset_path,
                embed_dataset_path=cfg.valid_embed_dataset_path,
                image_size=cfg.image_size,
            )
            self.valid_dataloader = DataLoader(
                self.valid_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,  # evaluate all data
            )
            self.valid_dataloader.collate_fn = iclevr_propv1_eval_collate_fn
            # test
            self.test_dataset = ICLEVRPropV1EvalDataset(
                dataset_path=cfg.test_dataset_path,
                embed_dataset_path=cfg.test_embed_dataset_path,
                image_size=cfg.image_size,
            )
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,  # evaluate all data
            )
            self.test_dataloader.collate_fn = iclevr_propv1_eval_collate_fn
        else:
            raise ValueError

    def evaluation(self, split="valid"):
        if split == "valid":
            eval_dataloader = self.valid_dataloader
            eval_dataset_path = self.cfg.valid_dataset_path
        elif split == "test":
            eval_dataloader = self.test_dataloader
            eval_dataset_path = self.cfg.test_dataset_path
        else:
            raise ValueError

        # create directory for latest (not best!) images
        dirname = os.path.join(SAVE_DIR, f"images_{split}/")
        os.makedirs(dirname, exist_ok=True)

        # generate images
        for batch in eval_dataloader:
            self.model.predict_batch(batch, dirname)

        # submission
        ap, ar, f1, _, rsim, cmat = \
            calculate_inception_objects_accuracy(
                dirname,
                self.cfg.detector_localizer_path,
                eval_dataset_path,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_workers,
            )

        outputs = {
            "scalar": {
                f"{split}_AP": ap,
                f"{split}_AR": ar,
                f"{split}_F1": f1,
                f"{split}_RSIM": rsim,
            },
            "others": {
                f"{split}_CMAT": cmat,
            }
        }
        return outputs

    def train(self):
        cnt_iter = 0

        for epoch in range(self.cfg.epochs):
            for batch in self.dataloader:
                cnt_iter += 1
                outputs = self.model.train_batch(batch)

                # evaluation
                if cnt_iter % self.cfg.eval_step == 0:
                    logger.debug("evaluate by valid data...")
                    val_outputs = self.evaluation(split="valid")
                    outputs["scalar"].update(val_outputs["scalar"])
                    outputs["others"].update(val_outputs["others"])

                    logger.debug("evaluate by test data...")
                    test_outputs = self.evaluation(split="test")
                    outputs["scalar"].update(test_outputs["scalar"])
                    outputs["others"].update(test_outputs["others"])

                # wandb upload
                if cnt_iter % self.cfg.vis_step == 0:
                    logger.debug(
                        (
                            f"epoch: {epoch}, step: {cnt_iter}, "
                            f"data: {outputs['scalar']}"
                        )
                    )
                    # self._upload_wandb(
                    #     outputs,
                    #     step=cnt_iter,
                    #     with_sample=(cnt_iter % self.cfg.vis_sample_step == 0),
                    # )

                # save model
                if cnt_iter % self.cfg.save_step == 0:
                    pass
                    # TODO: impl
                    # saved_path = self._save_model(epoch, cnt_iter)
                    # logger.info(
                    #     f"[e{epoch}, i{cnt_iter}] snapshot {saved_path} saved.")

    # def _upload_wandb(self, outputs, step, with_sample=False):
    #     log = {}

    #     # scalars, including AP, AR, F1, RSIM
    #     log.update(outputs["scalar"])

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

    #     try:
    #         # get other metrics (cmat)
    #         for k, v in outputs["others"].items():
    #             # case1. plot confusion matrix
    #             # only plot test confusion matrix
    #             if ("test_CMAT" in k) and (self.cfg.dataset == "codraw"):
    #                 # (58, 2, 2) -> (6, 10, 2, 2) including 2 blank plots
    #                 _, impath = plot_multilabel_confusion_matrix(
    #                     v, CODRAW_OBJS,
    #                     save_path=os.path.join(self.save_path, 'cmat'),
    #                     fname=f'cmat-{step}.png'
    #                 )
    #                 log.update({f'{k}_vis': wandb.Image(impath)})
    #     except KeyError:
    #         pass

    #     # upload
    #     wandb.log(log, step=step)


def train_propv1_geneva(cfg):
    logger.info(f"script {__name__} start!")

    # SAVE_DIR SETTING
    global SAVE_DIR
    SAVE_DIR = os.path.join(SAVE_ROOT_DIR, cfg.name)
    if os.path.exists(SAVE_DIR):
        logger.warning(f"{SAVE_DIR} already exists. Overwrite by current run?")

        if "stdin" in cfg:
            stdin = cfg.stdin
        else:
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
    trainer = PropV1Trainer(cfg)
    trainer.train()
