import os
import shutil
import uuid

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from modules.retrieval.model import TIRGModel
from data.codraw_retrieval_dataset import CoDrawRetrievalDataset
from data.codraw_retrieval_dataset import codraw_retrieval_collate_fn
from data.iclevr_retrieval_dataset import ICLEVRRetrievalDataset
from data.iclevr_retrieval_dataset import iclevr_retrieval_collate_fn
from utils.tracker import AverageMeter

# import wandb

from logging import getLogger
logger = getLogger(__name__)


plt.style.use("ggplot")

SAVE_ROOT_DIR = "./results/experiments/"
SAVE_DIR = None


class Trainer:
    def __init__(self, cfg):
        if cfg.dataset == "codraw":
            cfg.num_objects = 58
        elif cfg.dataset == "iclevr":
            cfg.num_objects = 24
        else:
            raise ValueError

        if "with_spade" not in cfg:
            cfg.with_spade = False

        self.cfg = cfg

        # trainer
        self.model = TIRGModel(
            depth=cfg.depth,
            sentence_encoder_type=cfg.sentence_encoder_type,
            with_spade=cfg.with_spade,
            text_dim=cfg.text_dim,
            hidden_dim=cfg.hidden_dim,
            sa_fused=cfg.sa_fused,
            sta_concat=cfg.sta_concat,
            use_pos_emb=cfg.use_pos_emb,
            sa_gate=cfg.sa_gate,
            res_mask=cfg.res_mask,
            res_mask_post=cfg.res_mask_post,
            use_conv_final=cfg.use_conv_final,
            multi_channel_gate=cfg.multi_channel_gate,
            optimizer_type=cfg.optimizer_type,
            lr=cfg.lr,
            lr_bert=cfg.lr_bert if "lr_bert" in cfg.keys() else 2e-5,
            weight_decay=cfg.weight_decay,
            loss_type=cfg.loss_type,
            margin=cfg.margin,
            k=cfg.k,
            gate_loss_gamma=cfg.gate_loss_gamma,
            text_detection_gamma=cfg.text_detection_gamma,
            gate_detection_gamma=cfg.gate_detection_gamma,
            num_objects=cfg.num_objects,
        )

        # dataset
        if cfg.dataset == "codraw":
            # codraw-train
            self.dataset = CoDrawRetrievalDataset(
                cfg.dataset_path,
                cfg.glove_path,
            )
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            self.dataloader.collate_fn = codraw_retrieval_collate_fn
            # codraw-valid
            self.valid_dataset = CoDrawRetrievalDataset(
                cfg.valid_dataset_path,
                cfg.glove_path,
            )
            self.valid_dataset.shuffle(cfg.seed)
            self.valid_dataloader = DataLoader(
                self.valid_dataset,
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,

            )
            self.valid_dataloader.collate_fn = codraw_retrieval_collate_fn
            # codraw-test
            self.test_dataset = CoDrawRetrievalDataset(
                cfg.test_dataset_path,
                cfg.glove_path,
            )
            self.test_dataset.shuffle(cfg.seed)
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
            self.test_dataloader.collate_fn = codraw_retrieval_collate_fn
        elif cfg.dataset == "iclevr":
            # iclevr-train
            self.dataset = ICLEVRRetrievalDataset(
                cfg.dataset_path,
                cfg.glove_path,
            )
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            self.dataloader.collate_fn = iclevr_retrieval_collate_fn
            # iclevr-valid
            self.valid_dataset = ICLEVRRetrievalDataset(
                cfg.valid_dataset_path,
                cfg.glove_path,
            )
            self.valid_dataloader = DataLoader(
                self.valid_dataset,
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
            self.valid_dataloader.collate_fn = iclevr_retrieval_collate_fn
            # iclevr-test
            self.test_dataset = ICLEVRRetrievalDataset(
                cfg.test_dataset_path,
                cfg.glove_path,
            )
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
            self.test_dataloader.collate_fn = iclevr_retrieval_collate_fn
        else:
            raise NotImplementedError

    def train(self):
        best_score = np.inf
        best_epoch = 0
        # best_state_dict_path = None

        for epoch in range(self.cfg.epochs):
            # train
            train_losses = AverageMeter()

            for batch in tqdm(self.dataloader):
                loss = self.model.train_batch(batch)
                train_losses.update(loss)

            # valid
            valid_losses = AverageMeter()
            if self.cfg.loss_type == "ranking":
                valid_sim_corrects = []
                valid_sim_wrongs = []
                valid_sim_bases = []
            elif self.cfg.loss_type == "l1":
                valid_l1_corrects = []
                valid_l1_bases = []
            if self.cfg.text_detection_gamma > 0.0:
                valid_f1_text = []
            if self.cfg.gate_detection_gamma > 0.0:
                valid_f1_gate = []

            for batch in self.valid_dataloader:
                outputs = self.model.predict_batch(batch)

                loss = outputs["loss"]
                rc = outputs["retrieval_correct"]
                rw = outputs["retrieval_wrong"]
                rs = outputs["retrieval_source"]
                l1c = outputs["l1loss_correct"]
                l1s = outputs["l1loss_source"]

                valid_losses.update(loss)
                if self.cfg.loss_type == "ranking":
                    valid_sim_corrects.append(rc)
                    valid_sim_wrongs.append(rw)
                    valid_sim_bases.append(rs)
                elif self.cfg.loss_type == "l1":
                    valid_l1_corrects.append(l1c)
                    valid_l1_bases.append(l1s)
                if "f1_text" in outputs.keys():
                    valid_f1_text.append(outputs["f1_text"])
                if "f1_gate" in outputs.keys():
                    valid_f1_gate.append(outputs["f1_gate"])

            if self.cfg.loss_type == "ranking":
                valid_sim_corrects = np.concatenate(valid_sim_corrects, axis=0)
                valid_sim_wrongs = np.concatenate(valid_sim_wrongs, axis=0)
                valid_sim_bases = np.concatenate(valid_sim_bases, axis=0)
            elif self.cfg.loss_type == "l1":
                valid_l1_corrects = np.concatenate(valid_l1_corrects, axis=0)
                valid_l1_bases = np.concatenate(valid_l1_bases, axis=0)

            if self.cfg.text_detection_gamma > 0.0:
                valid_f1_text = np.mean(valid_f1_text)
            if self.cfg.gate_detection_gamma > 0.0:
                valid_f1_gate = np.mean(valid_f1_gate)

            # test
            test_losses = AverageMeter()
            if self.cfg.loss_type == "ranking":
                test_sim_corrects = []
                test_sim_wrongs = []
                test_sim_bases = []
            elif self.cfg.loss_type == "l1":
                test_l1_corrects = []
                test_l1_bases = []
            if self.cfg.text_detection_gamma > 0.0:
                test_f1_text = []
            if self.cfg.gate_detection_gamma > 0.0:
                test_f1_gate = []

            for batch in self.test_dataloader:
                outputs = self.model.predict_batch(batch)

                loss = outputs["loss"]
                rc = outputs["retrieval_correct"]
                rw = outputs["retrieval_wrong"]
                rs = outputs["retrieval_source"]
                l1c = outputs["l1loss_correct"]
                l1s = outputs["l1loss_source"]

                test_losses.update(loss)
                if self.cfg.loss_type == "ranking":
                    test_sim_corrects.append(rc)
                    test_sim_wrongs.append(rw)
                    test_sim_bases.append(rs)
                elif self.cfg.loss_type == "l1":
                    test_l1_corrects.append(l1c)
                    test_l1_bases.append(l1s)
                if "f1_text" in outputs.keys():
                    test_f1_text.append(outputs["f1_text"])
                if "f1_gate" in outputs.keys():
                    test_f1_gate.append(outputs["f1_gate"])

            # get masks of the last batch
            mask_tup = self.model.get_curr_masks()
            image_tup = (
                batch['prev_image'].cpu().numpy(),
                batch['image'].cpu().numpy()
            )

            if self.cfg.loss_type == "ranking":
                test_sim_corrects = np.concatenate(test_sim_corrects, axis=0)
                test_sim_wrongs = np.concatenate(test_sim_wrongs, axis=0)
                test_sim_bases = np.concatenate(test_sim_bases, axis=0)
            elif self.cfg.loss_type == "l1":
                test_l1_corrects = np.concatenate(test_l1_corrects, axis=0)
                test_l1_bases = np.concatenate(test_l1_bases, axis=0)

            if self.cfg.text_detection_gamma > 0.0:
                test_f1_text = np.mean(test_f1_text)
            if self.cfg.gate_detection_gamma > 0.0:
                test_f1_gate = np.mean(test_f1_gate)

            # wandb logging

            loss_name = (
                "pairwise_ranking" if self.cfg.loss_type == "ranking"
                else self.cfg.loss_type
            )
            log = {
                f"{loss_name}_loss": train_losses.avg,
                f"valid_{loss_name}_loss": valid_losses.avg,
                f"test_{loss_name}_loss": test_losses.avg,
            }
            if self.cfg.text_detection_gamma > 0.0:
                log["valid_f1_text"] = valid_f1_text
                log["test_f1_text"] = test_f1_text
            if self.cfg.gate_detection_gamma > 0.0:
                log["valid_f1_gate"] = valid_f1_gate
                log["test_f1_gate"] = test_f1_gate

            # TODO: move this to utils
            # plot similarity distributions
            if self.cfg.loss_type == "ranking":
                fig, axes = plt.subplots(1, 2)
                fig.set_size_inches(16, 4)
                sns.kdeplot(
                    valid_sim_corrects, fill=True, label="correct", ax=axes[0])
                sns.kdeplot(
                    valid_sim_wrongs, fill=True, label="wrong", ax=axes[0])
                sns.kdeplot(
                    valid_sim_bases, fill=True, label="base", ax=axes[0])
                axes[0].legend()
                axes[0].set_title("valid")
                sns.kdeplot(
                    test_sim_corrects, fill=True, label="correct", ax=axes[1])
                sns.kdeplot(
                    test_sim_wrongs, fill=True, label="wrong", ax=axes[1])
                sns.kdeplot(
                    test_sim_bases, fill=True, label="base", ax=axes[1])
                axes[1].legend()
                axes[1].set_title("test")
                # log.update({
                #     "feature similarity disribution plot": wandb.Image(plt),
                # })
            elif self.cfg.loss_type == "l1":
                # valid
                lim = max(valid_l1_bases.max(), valid_l1_corrects.max())
                valid_fig = sns.jointplot(
                    x=valid_l1_bases,
                    y=valid_l1_corrects,
                    marker="+",
                    xlim=(0., lim),
                    ylim=(0., lim),
                    marginal_ticks=True,
                )
                valid_fig.set_axis_labels(
                    xlabel="valid_l1_src_tgt",
                    ylabel="valid_l1_pred_tgt",
                )
                x0, x1 = valid_fig.ax_joint.get_xlim()
                y0, y1 = valid_fig.ax_joint.get_ylim()
                lims = [max(x0, y0), min(x1, y1)]
                valid_fig.ax_joint.plot(lims, lims, ':k')
                # test
                lim = max(test_l1_bases.max(), test_l1_corrects.max())
                test_fig = sns.jointplot(
                    x=test_l1_bases,
                    y=test_l1_corrects,
                    marker="+",
                    xlim=(0., lim),
                    ylim=(0., lim),
                    marginal_ticks=True,
                )
                test_fig.set_axis_labels(
                    xlabel="test_l1_src_tgt",
                    ylabel="test_l1_pred_tgt",
                )
                x0, x1 = test_fig.ax_joint.get_xlim()
                y0, y1 = test_fig.ax_joint.get_ylim()
                lims = [max(x0, y0), min(x1, y1)]
                test_fig.ax_joint.plot(lims, lims, ':k')
                # wandb
                # log.update({
                #     "valid feature l1 loss scatter plot":
                #     wandb.Image(valid_fig.fig),
                #     "test feature l1 loss scatter plot":
                #     wandb.Image(test_fig.fig),
                # })

            # TODO: move the following to utils
            # plot residual masks
            _ = create_mask_visualization(*image_tup, *mask_tup)
            # log.update({"mask visualization": mask_logs})

            # log
            logger.info(f"epoch: {epoch}/{self.cfg.epochs}, losses: {log}")
            # wandb.log(log, step=epoch)

            # clear plot
            plt.cla()
            plt.clf()
            plt.close()

            # model saving
            if test_losses.avg < best_score:
                # best_state_dict_path = self._save_model(best_epoch, epoch)
                _ = self._save_model(best_epoch, epoch)
                best_score = test_losses.avg
                best_epoch = epoch

        # save best model on wandb
        # wandb.save(best_state_dict_path)

    def _save_model(self, prev_best_epoch, curr_epoch):
        # create directory for snapshots
        save_snapshot_dir = os.path.join(SAVE_DIR, "snapshots")
        os.makedirs(save_snapshot_dir, exist_ok=True)

        # save current best snapshot
        snapshot = self.model.get_state_dict()
        curr_path = os.path.join(
            save_snapshot_dir,
            f"{self.cfg.dataset}_pretrain_models_epoch_{curr_epoch}.pth",
        )
        torch.save(snapshot, curr_path)
        logger.info(f"snapshot saved: {curr_path}")

        # delete previous best snapshot for device space
        if prev_best_epoch != curr_epoch:
            prev_path = os.path.join(
                save_snapshot_dir,
                f"{self.cfg.dataset}_pretrain_models_epoch_{prev_best_epoch}.pth",
            )
            os.remove(prev_path)
            logger.info(f"previous best snapshot deleted: {prev_path}")

        return curr_path


def create_mask_visualization(
    prev_img, curr_img, mask_gt, mask_pred, batch_ids=[0, 1, 2], for_wandb=False
):
    """create_mask_visualization.

    Parameters
    ----------
    prev_img : (B, C, H, W)
        prev_img
    curr_img : (B, C, H, W)
        curr_img
    mask_gt : (B, 1, H, W)
        mask_gt
    mask_pred : (B, 1, H, W)
        mask_pred
    batch_ids : list
        batch indecies that are to be saved.
    for_wandb : bool
        true then convert images to wandb format.
    """

    imgs = []
    for b in batch_ids:
        # images
        # pi, ci.shape: (C, H, W) -> (H, W, C)
        pi = prev_img[b].transpose((1, 2, 0))
        ci = curr_img[b].transpose((1, 2, 0))
        pi, ci = (pi + 1) * 128, (ci + 1) * 128
        pi, ci = pi.astype(np.uint8), ci.astype(np.uint8)
        pi, ci = np.clip(pi, 0, 255), np.clip(ci, 0, 255)
        h, w, c = pi.shape

        # masks: (C, H, W) -> (H, W, C)
        mg = mask_gt[b].transpose((1, 2, 0))
        mp = mask_pred[b].transpose((1, 2, 0))

        # upsample
        ratio = (h // mg.shape[0])
        mg = mg.repeat(ratio, axis=0).repeat(ratio, axis=1)
        mp = mp.repeat(ratio, axis=0).repeat(ratio, axis=1)

        # pad shape
        _h, _w, _ = mg.shape
        pad_h, pad_w = h - _h, w - _w
        pad_shape = ((0, pad_h), (0, pad_w), (0, 0))
        mg = np.pad(mg, pad_shape, 'constant', constant_values=0)
        mp = np.pad(mp, pad_shape, 'constant', constant_values=0)
        mg, mp = np.tile(mg, 3) * 255, np.tile(mp, 3) * 255
        mg, mp = mg.astype(np.uint8), mp.astype(np.uint8)

        # margin: (H, 5, C)
        _mar = np.zeros((h, 5, c))

        # concat (H, W+W+5*3, C)
        cct = np.concatenate((pi, _mar, ci, _mar, mg, _mar, mp), axis=1)

        if for_wandb:
            raise NotImplementedError
            # cct = wandb.Image(cct, caption=f'Mask Visualization ({b})')

        imgs.append(cct)

    return imgs


def pretrain_tirg(cfg):
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

    trainer = Trainer(cfg)
    trainer.train()
