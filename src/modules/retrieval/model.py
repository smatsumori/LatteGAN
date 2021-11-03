import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18

from transformers import BertTokenizerFast
from adabelief_pytorch import AdaBelief

from .sentence_encoder import SentenceEncoder, BERTSentenceEncoder
from .tirg import TIRG
from .tirg_spade import TIRGSPADE


class TIRGModel:
    def __init__(
        self,
        depth=5,
        sentence_encoder_type="rnn",
        with_spade=False,
        text_dim=512,
        hidden_dim=512,
        sa_fused=False,
        sta_concat=False,
        use_pos_emb=False,
        sa_gate=False,
        res_mask=False,
        res_mask_post=False,
        use_conv_final=False,
        multi_channel_gate=False,
        optimizer_type="adam",
        lr=0.001,
        lr_bert=2e-5,
        weight_decay=0.0001,
        loss_type="ranking",
        margin=1.0,
        k=1,
        gate_loss_gamma=0.0,
        text_detection_gamma=0.0,
        gate_detection_gamma=0.0,
        num_objects=58,
    ):
        # models

        n = resnet18(pretrained=True)
        idx = depth - 5 - 2
        n = list(n.children())[:idx]
        self.cnn = nn.DataParallel(nn.Sequential(*n)).cuda()
        self.channels = 512 // (2 ** (5 - depth))
        # freeze cnn
        self.cnn.eval()
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.sentence_encoder_type = sentence_encoder_type
        if sentence_encoder_type == "rnn":
            self.rnn_prev = nn.DataParallel(SentenceEncoder(text_dim)).cuda()
            self.rnn_curr = nn.DataParallel(SentenceEncoder(text_dim)).cuda()
        elif sentence_encoder_type == "bert":
            self.tokenizer = BertTokenizerFast.\
                from_pretrained("bert-base-uncased")
            self.bert = nn.DataParallel(BERTSentenceEncoder()).cuda()
        else:
            raise ValueError
        # sentence vector to detection
        self.detector = nn.DataParallel(
            nn.Linear(text_dim, num_objects)).cuda()

        self.multi_channel_gate = multi_channel_gate
        if with_spade:
            self.tirg = nn.DataParallel(TIRGSPADE(
                self.channels,
                text_dim,
                hidden_dim,
                sta_concat=sta_concat,
                use_pos_emb=use_pos_emb,
                res_mask_post=res_mask_post,
                multi_channel_gate=multi_channel_gate,
                num_objects=num_objects,
            )).cuda()
        else:
            self.tirg = nn.DataParallel(TIRG(
                self.channels,
                text_dim,
                hidden_dim,
                sa_fused=sa_fused,
                sta_concat=sta_concat,
                use_pos_emb=use_pos_emb,
                sa_gate=sa_gate,
                res_mask=res_mask,
                res_mask_post=res_mask_post,
                use_conv_final=use_conv_final,
                multi_channel_gate=multi_channel_gate,
            )).cuda()

        # optimizers

        kwargs = {}
        if optimizer_type == "adam":
            optimizer = torch.optim.Adam
        elif optimizer_type == "adamw":
            optimizer = torch.optim.AdamW
        elif optimizer_type == "adabelief":
            optimizer = AdaBelief
            kwargs["weight_decouple"] = False
            kwargs["rectify"] = False
        else:
            raise ValueError

        if sentence_encoder_type == "rnn":
            params = list(self.tirg.parameters())
            params += list(self.rnn_prev.parameters())
            params += list(self.rnn_curr.parameters())
            self.optimizer = optimizer(
                params,
                lr=lr,
                weight_decay=weight_decay,
                **kwargs,
            )
        elif sentence_encoder_type == "bert":
            self.optimizer = optimizer(
                [
                    {"params": self.tirg.parameters()},
                    {"params": self.bert.parameters(),
                     "lr": lr_bert,
                     "weight_decay": 0.0},
                ],
                lr=lr,
                weight_decay=weight_decay,
                **kwargs,
            )

        # criterion
        self.loss_type = loss_type
        if loss_type == "ranking":
            self.pool = nn.DataParallel(nn.AdaptiveAvgPool2d(1)).cuda()
            self.cossim = nn.DataParallel(nn.CosineSimilarity(dim=1)).cuda()
            self.relu = nn.DataParallel(nn.ReLU()).cuda()
            self.margin = margin
            self.k = k
        elif loss_type == "l1":
            self.l1 = nn.DataParallel(nn.L1Loss(reduction="none")).cuda()
        else:
            raise ValueError

        # gate loss
        self.bce = nn.DataParallel(nn.BCELoss(reduction="none")).cuda()
        self.gate_loss_gamma = gate_loss_gamma

        # detection loss
        self.bce_logits = nn.DataParallel(
            nn.BCEWithLogitsLoss(reduction="none")).cuda()
        self.text_detection_gamma = text_detection_gamma
        self.gate_detection_gamma = gate_detection_gamma

        # placeholders
        self.pred_gate = None
        self.noisy_true_gate = None

    def predict_batch(self, batch):
        if self.sentence_encoder_type == "rnn":
            self.rnn_prev.eval()
            self.rnn_curr.eval()
        elif self.sentence_encoder_type == "bert":
            self.bert.eval()
        self.detector.eval()

        self.tirg.eval()
        if self.loss_type == "ranking":
            self.pool.eval()
            self.cossim.eval()
            self.relu.eval()
        elif self.loss_type == "l1":
            self.l1.eval()

        with torch.no_grad():
            prev_image = batch["prev_image"]
            image = batch["image"]

            prev_image_map = self.cnn(prev_image)
            image_map = self.cnn(image)

            text_memories = None
            text_feature = None
            seq_len = None
            key_padding_mask = None

            if self.sentence_encoder_type == "rnn":
                # extract from batch
                prev_embs = batch["prev_embs"]
                embs = batch["embs"]
                prev_seq_len = batch["prev_seq_len"]
                seq_len = batch["seq_len"]
                # forward sentence encoder
                _, _, context = self.rnn_prev(prev_embs, prev_seq_len)
                text_memories, text_feature, _ = self.rnn_curr(
                    embs, seq_len, context)
            elif self.sentence_encoder_type == "bert":
                # extract from batch
                prev_utter = batch["prev_utter"]
                utter = batch["utter"]
                # tokenize
                inputs = self.tokenizer(
                    text=prev_utter,
                    text_pair=utter,
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                # forward sentence encoder
                text_memories, text_feature, key_padding_mask = \
                    self.bert(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        inputs["token_type_ids"],
                    )

            # added object detection from sentence vector
            if self.text_detection_gamma > 0.0:
                objs = batch["added_objects"]

                pred_objs = self.detector(text_feature)
                pred_objs = torch.sigmoid(pred_objs) > 0.5

                objs = objs.cpu().numpy().astype(np.int64)
                pred_objs = pred_objs.cpu().numpy().astype(np.int64)
                f1_text_detect = f1_score(
                    objs,
                    pred_objs,
                    average="samples",
                    zero_division=0,
                )

            # store pred_gate for visualization
            pred_image_map, self.pred_gate, pred_objs_gate = self.tirg(
                prev_image_map,
                text_feature,
                text_memories,
                seq_len,
                key_padding_mask,
            )

            # added object detection from gate pass
            if self.gate_detection_gamma > 0.0:
                objs = batch["added_objects"]

                pred_objs_gate = torch.sigmoid(pred_objs_gate) > 0.5

                objs = objs.cpu().numpy().astype(np.int64)
                pred_objs_gate = pred_objs_gate.cpu().numpy().astype(np.int64)
                f1_gate_detect = f1_score(
                    objs,
                    pred_objs_gate,
                    average="samples",
                    zero_division=0,
                )

            # compute noisy_true_gate for visualization
            self.noisy_true_gate = self._get_noisy_true_gate(
                prev_image_map, image_map
            )

            loss, rc, rw, rs, l1c, l1s = self._network_loss(
                pred_image_map,
                image_map,
                prev_image_map,
            )
            loss = loss.item()

        if self.sentence_encoder_type == "rnn":
            self.rnn_prev.train()
            self.rnn_curr.train()
        elif self.sentence_encoder_type == "bert":
            self.bert.train()
        self.detector.train()

        self.tirg.train()
        if self.loss_type == "ranking":
            self.pool.train()
            self.cossim.train()
            self.relu.train()
        elif self.loss_type == "l1":
            self.l1.train()

        outputs = {
            "loss": loss,
            "retrieval_correct": rc,
            "retrieval_wrong": rw,
            "retrieval_source": rs,
            "l1loss_correct": l1c,
            "l1loss_source": l1s,
        }
        if self.text_detection_gamma > 0.0:
            outputs["f1_text"] = f1_text_detect
        if self.gate_detection_gamma > 0.0:
            outputs["f1_gate"] = f1_gate_detect

        return outputs

    def train_batch(self, batch):
        prev_image = batch["prev_image"]
        image = batch["image"]

        prev_image_map = self.cnn(prev_image)
        image_map = self.cnn(image)
        noisy_true_gate = self._get_noisy_true_gate(
            prev_image_map, image_map
        )

        text_memories = None
        text_feature = None
        seq_len = None
        key_padding_mask = None

        if self.sentence_encoder_type == "rnn":
            # extract from batch
            prev_embs = batch["prev_embs"]
            embs = batch["embs"]
            prev_seq_len = batch["prev_seq_len"]
            seq_len = batch["seq_len"]
            # forward sentence encoder
            _, _, context = self.rnn_prev(prev_embs, prev_seq_len)
            text_memories, text_feature, _ = self.rnn_curr(
                embs, seq_len, context)
        elif self.sentence_encoder_type == "bert":
            # extract from batch
            prev_utter = batch["prev_utter"]
            utter = batch["utter"]
            # tokenize
            inputs = self.tokenizer(
                text=prev_utter,
                text_pair=utter,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            # forward sentence encoder
            text_memories, text_feature, key_padding_mask = \
                self.bert(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    inputs["token_type_ids"],
                )

        # added object detection from sentence vector
        added_objects = batch["added_objects"]
        pred_added_objects = self.detector(text_feature)

        pred_image_map, pred_gate, pred_added_objects_gate = self.tirg(
            prev_image_map,
            text_feature,
            text_memories,
            seq_len,
            key_padding_mask,
        )

        # loss & backward & step
        loss = self._optimize_network(
            pred_image_map,
            image_map,
            pred_gate,
            noisy_true_gate,
            pred_added_objects,
            added_objects,
            pred_added_objects_gate,
        )

        return loss

    def get_curr_masks(self):
        """ Returns masks for the current batch.
        """
        # g, p: (B, C, H, W)
        g = self.noisy_true_gate.cpu().numpy()
        p = self.pred_gate.cpu().numpy()
        if g.shape[1] > 1:
            # multi channel setting
            g = g.mean(axis=1, keepdims=True)
            p = p.mean(axis=1, keepdims=True)

        return g, p

    def _get_noisy_true_gate(self, h_src, h_tgt, eps=1e-16):
        m = torch.abs(h_src - h_tgt)
        if not self.multi_channel_gate:
            m = m.mean(dim=1, keepdims=True)  # shape=(B, 1, H, W)
        m = m / (m.max() + eps)  # range=(0, 1)
        g = 1 - m
        return g

    def _optimize_network(
        self, v, u, pred_gate, true_gate, pred_objs, true_objs, pred_objs_gate
    ):
        with torch.autograd.detect_anomaly():
            self.optimizer.zero_grad()

            loss, _, _, _, _, _ = self._network_loss(v, u)
            if self.gate_loss_gamma > 0.0:
                loss += self.gate_loss_gamma * \
                    self._network_gate_loss(pred_gate, true_gate)
            if self.text_detection_gamma > 0.0:
                loss += self.text_detection_gamma * \
                    self._network_detect_loss(pred_objs, true_objs)
            if self.gate_detection_gamma > 0.0:
                loss += self.gate_detection_gamma * \
                    self._network_detect_loss(pred_objs_gate, true_objs)

            loss.backward()
            self.optimizer.step()

        return loss.item()

    def _network_detect_loss(self, po, to):
        loss = self.bce_logits(po, to).mean()

        return loss

    def _network_gate_loss(self, pg, tg):
        loss = self.bce(pg, tg).mean()

        return loss

    def _network_loss(self, v, u, v_src=None):
        loss = 0
        rc = None
        rw = None
        rs = None
        l1c = None
        l1s = None

        if self.loss_type == "ranking":
            # pooling image map
            v = self.pool(v).view(-1, self.channels)
            u = self.pool(u).view(-1, self.channels)

            # calculate gt pair cossim
            rc = self.cossim(v, u)

            loss = 0
            rw_ls = []
            for i in range(1, self.k + 1):
                v_wrong = torch.cat((v[i:], v[0:i]), dim=0)
                u_wrong = torch.cat((u[i:], u[0:i]), dim=0)

                rw1 = self.cossim(v, u_wrong)
                rw2 = self.cossim(v_wrong, u)

                loss += self.relu(self.margin - rc + rw1).mean() / self.k
                loss += self.relu(self.margin - rc + rw2).mean() / self.k

                rw = torch.cat((rw1.detach(), rw2.detach()), dim=0)
                rw_ls.append(rw)

            rc = rc.detach().cpu().numpy()
            rw = torch.cat(rw_ls, dim=0).cpu().numpy()

            if v_src is not None:
                v_src = self.pool(v_src).view(-1, self.channels)
                rs = self.cossim(u, v_src)
                rs = rs.detach().cpu().numpy()

        elif self.loss_type == "l1":
            l1c = self.l1(v, u).mean(dim=(1, 2, 3))
            loss = l1c.mean()

            l1c = l1c.detach().cpu().numpy()
            if v_src is not None:
                l1s = self.l1(v_src, u).mean(dim=(1, 2, 3))
                l1s = l1s.detach().cpu().numpy()

        else:
            raise NotImplementedError

        return loss, rc, rw, rs, l1c, l1s

    def get_state_dict(self):
        state_dict = {}
        if self.sentence_encoder_type == "rnn":
            state_dict["rnn_prev"] = self.rnn_prev.state_dict()
            state_dict["rnn_curr"] = self.rnn_curr.state_dict()
        elif self.sentence_encoder_type == "bert":
            state_dict["bert"] = self.bert.state_dict()

        return state_dict
