import os
from collections import defaultdict

import h5py
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BertTokenizerFast

from data.codraw_retrieval_dataset import CoDrawRetrievalDataset, codraw_retrieval_collate_fn
from data.iclevr_retrieval_dataset import ICLEVRRetrievalDataset, iclevr_retrieval_collate_fn
from modules.retrieval.sentence_encoder import SentenceEncoder, BERTSentenceEncoder

from logging import getLogger
logger = getLogger(__name__)


class SentenceEmbeddingGenerator:
    def __init__(self, cfg):
        self.cfg = cfg

        # model
        self.sentence_encoder_type = cfg.sentence_encoder_type
        if "model_path" not in cfg:
            logger.warning(
                "model_path is not specified. "
                "use initial weight of pretrained models."
            )
            state_dict = None
        else:
            state_dict = torch.load(cfg.model_path)

        if cfg.sentence_encoder_type == "rnn":
            self.rnn_prev = nn.DataParallel(
                SentenceEncoder(cfg.text_dim),
                device_ids=[0],
            ).cuda()
            self.rnn_curr = nn.DataParallel(
                SentenceEncoder(cfg.text_dim),
                device_ids=[0],
            ).cuda()
            if state_dict is not None:
                self.rnn_prev.load_state_dict(state_dict["rnn_prev"])
                self.rnn_curr.load_state_dict(state_dict["rnn_curr"])
            self.rnn_prev.eval()
            self.rnn_curr.eval()
        elif cfg.sentence_encoder_type == "bert":
            self.tokenizer = BertTokenizerFast.\
                from_pretrained("bert-base-uncased")
            self.bert = nn.DataParallel(
                BERTSentenceEncoder(),
                device_ids=[0],
            ).cuda()
            if state_dict is not None:
                self.bert.load_state_dict(state_dict["bert"])
            self.bert.eval()
        else:
            raise ValueError

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
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
            self.dataloader.collate_fn = codraw_retrieval_collate_fn
            # codraw-valid
            self.valid_dataset = CoDrawRetrievalDataset(
                cfg.valid_dataset_path,
                cfg.glove_path,
            )
            self.valid_dataloader = DataLoader(
                self.valid_dataset,
                batch_size=cfg.batch_size,
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
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=cfg.batch_size,
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
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
            self.dataloader.collate_fn = iclevr_retrieval_collate_fn
            # iclevr-valid
            self.valid_dataset = ICLEVRRetrievalDataset(
                cfg.valid_dataset_path,
                cfg.glove_path,
            )
            self.valid_dataloader = DataLoader(
                self.valid_dataset,
                batch_size=cfg.batch_size,
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
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
            self.test_dataloader.collate_fn = iclevr_retrieval_collate_fn
        else:
            raise ValueError

    def generate(self, save_path, split="train"):
        # keys = [(access_id, turn_index), ...]
        if split == "train":
            keys = self.dataset.keys
            dataloader = self.dataloader
        elif split == "valid":
            keys = self.valid_dataset.keys
            dataloader = self.valid_dataloader
        elif split == "test":
            keys = self.test_dataset.keys
            dataloader = self.test_dataloader
        else:
            raise ValueError

        # all_text_features: List of Tensor (D,)
        # all_text_memories: List of Tensor (ml, D)
        # all_text_lengths: List of int [text_length, ...]
        # mb: mini-batch, ml: max dialog length in mini-batch
        all_text_features = []
        all_text_memories = []
        all_text_lengths = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                if self.sentence_encoder_type == "rnn":
                    # extract from batch
                    prev_embs = batch["prev_embs"]
                    embs = batch["embs"]
                    prev_seq_len = batch["prev_seq_len"]
                    seq_len = batch["seq_len"]

                    # forward sentence encoder
                    _, _, context = self.rnn_prev(
                        prev_embs, prev_seq_len)
                    text_memories, text_feature, _ = self.rnn_curr(
                        embs, seq_len, context)

                elif self.sentence_encoder_type == "bert":
                    prev_utter = batch["prev_utter"]
                    utter = batch["utter"]

                    inputs = self.tokenizer(
                        text=prev_utter,
                        text_pair=utter,
                        add_special_tokens=True,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )

                    text_memories, text_feature, key_padding_mask = self.bert(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        inputs["token_type_ids"],
                    )

                # push to list
                for i in range(text_memories.size(0)):
                    if self.sentence_encoder_type == "rnn":
                        all_text_features.append(
                            text_feature[i].cpu().numpy())
                        all_text_memories.append(
                            text_memories[i].cpu().numpy())
                        all_text_lengths.append(
                            seq_len[i].cpu().numpy())
                    elif self.sentence_encoder_type == "bert":
                        all_text_features.append(
                            text_feature[i].cpu().numpy())

                        # output text memories of bert includes embedding of previous instructions.
                        # key_padding_mask is False where [CLS] & [[SENTENCE B], ...] & [SEP]
                        # sum of NOT key_padding_mask == text length
                        bool_indices = ~key_padding_mask[i]
                        _seq_len = np.array(bool_indices.sum().item())
                        _text_memories = text_memories[i][bool_indices]

                        all_text_memories.append(
                            _text_memories.cpu().numpy())
                        all_text_lengths.append(
                            _seq_len)

        # mapping dataset_id(did) to tuple of (start_index, end_index)
        # keys = [(access_id, turn_index), ...]
        id2idxtup = defaultdict(list)
        for i, (did, tid) in enumerate(keys):
            if did not in id2idxtup:
                id2idxtup[did] = [i, i]
            else:
                id2idxtup[did][1] = i

        # create h5 datasets
        h5 = h5py.File(save_path, "w")
        for did in id2idxtup.keys():
            start, end = id2idxtup[did]
            end += 1

            # turns_text_embedding: shape=(l, D)
            # turns_text_length: shape=(l,)
            # l: dialog length
            turns_text_embedding = np.stack(
                all_text_features[start:end], axis=0)
            turns_text_length = np.array(
                all_text_lengths[start:end])

            # turns_word_embeddings: shape=(l, ms, D)
            # ms: max text length of a dialog
            # it means that turns_word_embeddings is already padded.
            turns_word_embeddings = np.zeros(
                (len(turns_text_length), max(turns_text_length), self.cfg.text_dim))
            for i, j in enumerate(range(start, end)):
                text_length = turns_text_length[i]
                turns_word_embeddings[i, :text_length] = \
                    all_text_memories[j][:text_length]

            scene = h5.create_group(did)
            scene.create_dataset(
                "turns_text_embedding", data=turns_text_embedding)
            scene.create_dataset(
                "turns_word_embeddings", data=turns_word_embeddings)
            scene.create_dataset(
                "turns_text_length", data=turns_text_length)


def create_embs_from_model(cfg):
    logger.info(f"script {__name__} start!")

    generator = SentenceEmbeddingGenerator(cfg)

    for split in ["train", "valid", "test"]:
        save_path = os.path.join(
            cfg.save_root_dir,
            f"{cfg.dataset}_{split}_embeddings_{cfg.fork}.h5",
        )
        logger.info(f"create additional dataset: {save_path}")
        generator.generate(save_path, split)
