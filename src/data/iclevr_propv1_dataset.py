from ast import literal_eval

import h5py
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


def _decode_text_from_h5(text):
    return literal_eval(literal_eval(str(text)).decode())


def _decode_scene_id_from_h5(scene_id):
    return literal_eval(str(scene_id)).decode()


def _image_preprocessing(image, height=128, width=128):
    shape = image.shape
    if len(shape) == 3:
        h, w, _ = shape
        transpose = (2, 0, 1)
    elif len(shape) == 4:
        _, h, w, _ = shape
        transpose = (0, 3, 1, 2)
    else:
        raise ValueError

    if (h != height) or (w != width):
        new_image = []
        if len(shape) == 3:
            image = image[None, :, :, :]
        for i in range(len(image)):
            new_image.append(cv2.resize(image[i], (height, width)))
        image = np.stack(new_image, axis=0)
        if len(shape) == 3:
            image = image[0]

    image = image[..., ::-1]
    image = image / 128. - 1
    image = np.transpose(image, transpose)

    return image


class ICLEVRPropV1TrainDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        embed_dataset_path,
        image_size=128,
        **kwargs,
    ):
        self.dataset_path = dataset_path
        self.embed_dataset_path = embed_dataset_path
        self.image_size = image_size

        self.dataset = None
        self.embed_dataset = None
        self.background = None
        # keys = [(access_id, turn_index), ...]
        self.keys = []

        with h5py.File(dataset_path, "r") as f:
            background = f["background"][...]
            self.background = cv2.resize(background, (128, 128))

            # _keys = [access_id, ...]
            # NOTE: access_id != scene_id, access_id is an access key of h5 to each episodes.
            # NOTE: f.keys() = [{access_id}, ..., background, entities]
            _keys = [key for key in f.keys() if key.isdigit()]
            for key in _keys:
                dialog_length = f[key]["objects"][...].shape[0]
                assert dialog_length == 5, \
                    f"iclevr dialog length must be 5 but {dialog_length} of {key}."
                self.keys.extend([(key, t) for t in range(dialog_length)])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.dataset_path, "r")
        if self.embed_dataset is None:
            self.embed_dataset = h5py.File(self.embed_dataset_path, "r")

        access_id, turn_index = self.keys[idx]
        example = self.dataset[access_id]
        image = example["images"][...][turn_index]  # (H, W, C), BGR
        objects = example["objects"][...][turn_index]  # (24,)
        # both text and scene_id need decode preprocessing
        utter = _decode_text_from_h5(example["text"][...])[turn_index]
        scene_id = _decode_scene_id_from_h5(example["scene_id"][...])

        if turn_index > 0:
            prev_image = example["images"][...][turn_index - 1]
            prev_objects = example["objects"][...][turn_index - 1]
        else:
            prev_image = self.background
            prev_objects = np.zeros_like(objects)

        # fetch embeddings of utter from generated dataset
        embed_example = self.embed_dataset[access_id]
        text_embedding = embed_example["turns_text_embedding"][...][turn_index]
        word_embeddings = embed_example["turns_word_embeddings"][...][turn_index]
        text_length = embed_example["turns_text_length"][...][turn_index]

        # image preprocessing
        image = _image_preprocessing(
            image, self.image_size, self.image_size)
        prev_image = _image_preprocessing(
            prev_image, self.image_size, self.image_size)

        # text preprocessing
        word_embeddings = word_embeddings[:text_length]

        # added objects binary flag
        added_objects = objects - prev_objects
        added_objects = np.clip(added_objects, 0, 1)

        sample = {
            "source_image": prev_image,
            "target_image": image,
            "text_embedding": text_embedding,
            "word_embeddings": word_embeddings,
            "text_length": text_length,
            "utter": utter,
            "objects": objects,
            "added_objects": added_objects,
            "scene_id": scene_id,
        }
        return sample


def iclevr_propv1_train_collate_fn(batch):
    batch_size = len(batch)
    c, h, w = batch[0]["source_image"].shape
    d = batch[0]["text_embedding"].shape[0]
    max_text_length = max([b["text_length"] for b in batch])

    # placeholders
    batch_source_image = np.zeros(
        (batch_size, c, h, w), dtype=np.float32)
    batch_target_image = np.zeros(
        (batch_size, c, h, w), dtype=np.float32)
    batch_text_embedding = np.zeros(
        (batch_size, d), dtype=np.float32)
    batch_word_embeddings = np.zeros(
        (batch_size, max_text_length, d), dtype=np.float32)
    batch_text_length = np.zeros(
        (batch_size,), dtype=np.int64)
    batch_objects = np.zeros(
        (batch_size, 24), dtype=np.float32)
    batch_added_objects = np.zeros(
        (batch_size, 24), dtype=np.float32)
    batch_utter = []

    for i, b in enumerate(batch):
        src_img = b["source_image"]
        tgt_img = b["target_image"]
        txt_emb = b["text_embedding"]
        wrd_embs = b["word_embeddings"]
        txt_len = b["text_length"]
        objs = b["objects"]
        ad_objs = b["added_objects"]

        batch_source_image[i] = src_img
        batch_target_image[i] = tgt_img
        batch_text_embedding[i] = txt_emb
        batch_word_embeddings[i, :txt_len] = wrd_embs
        batch_text_length[i] = txt_len
        batch_objects[i] = objs
        batch_added_objects[i] = ad_objs

        utr = b["utter"]
        batch_utter.append(utr)

    sample = {
        "source_image": torch.FloatTensor(batch_source_image),
        "target_image": torch.FloatTensor(batch_target_image),
        "text_embedding": torch.FloatTensor(batch_text_embedding),
        "word_embeddings": torch.FloatTensor(batch_word_embeddings),
        "text_length": torch.LongTensor(batch_text_length),
        "objects": torch.FloatTensor(batch_objects),
        "added_objects": torch.FloatTensor(batch_added_objects),
        "utter": batch_utter,
    }
    return sample


class ICLEVRPropV1EvalDataset:
    def __init__(
        self,
        dataset_path,
        embed_dataset_path,
        image_size=128,
        **kwargs,
    ):
        self.dataset_path = dataset_path
        self.embed_dataset_path = embed_dataset_path
        self.image_size = image_size

        self.dataset = None
        self.embed_dataset = None
        self.background = None
        # keys = [access_id, ...]
        self.keys = []

        with h5py.File(dataset_path, "r") as f:
            background = f["background"][...]
            background = cv2.resize(background, (128, 128))
            background = background[..., ::-1].transpose(2, 0, 1)
            self.background = background / 128. - 1

            self.keys = [key for key in f.keys() if key.isdigit()]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.dataset_path, "r")
        if self.embed_dataset is None:
            self.embed_dataset = h5py.File(self.embed_dataset_path, "r")

        access_id = self.keys[idx]
        example1 = self.dataset[access_id]
        images = example1["images"][...]  # (L, H, W, C), BGR
        # both text and scene_id need decode preprocessing
        utterences = _decode_text_from_h5(example1["text"][...])
        scene_id = _decode_scene_id_from_h5(example1["scene_id"][...])

        # fetch embeddings of utter from generated dataset
        example2 = self.embed_dataset[access_id]
        turns_text_embedding = \
            example2["turns_text_embedding"][...]  # (L, D)
        turns_word_embeddings = \
            example2["turns_word_embeddings"][...]  # (L, S, D)
        turns_text_length = \
            example2["turns_text_length"][...]  # (L,)

        # image preprocessing
        images = _image_preprocessing(
            images, self.image_size, self.image_size)

        sample = {
            "background": self.background,
            "turns_image": images,
            "turns_text_embedding": turns_text_embedding,
            "turns_word_embeddings": turns_word_embeddings,
            "turns_text_length": turns_text_length,
            "scene_id": scene_id,
            "turns_utterence": utterences,
        }
        return sample


def iclevr_propv1_eval_collate_fn(batch):
    fixed_dialog_length = 5  # iCLEVR num turns == 5
    dialog_lengths = list(map(lambda x: len(x["turns_image"]), batch))
    assert np.all([dlen == fixed_dialog_length for dlen in dialog_lengths])

    batch_max_text_length = [max(b["turns_text_length"]) for b in batch]
    max_text_length = max(batch_max_text_length)

    batch_size = len(batch)
    _, c, h, w = batch[0]["turns_image"].shape
    _, d = batch[0]["turns_text_embedding"].shape

    # placeholders
    batch_turns_image = np.zeros(
        (batch_size, fixed_dialog_length, c, h, w),
        dtype=np.float32,
    )
    batch_turns_text_embedding = np.zeros(
        (batch_size, fixed_dialog_length, d),
        dtype=np.float32,
    )
    batch_turns_word_embeddings = np.zeros(
        (batch_size, fixed_dialog_length, max_text_length, d),
        dtype=np.float32,
    )
    batch_turns_text_length = np.zeros(
        (batch_size, fixed_dialog_length),
        dtype=np.int64,
    )
    batch_scene_id = []
    batch_turns_utterence = []

    background = None
    for i, b in enumerate(batch):
        background = b["background"]

        turns_image = b["turns_image"]
        turns_text_embedding = b["turns_text_embedding"]
        turns_word_embeddings = b["turns_word_embeddings"]
        turns_text_length = b["turns_text_length"]

        tlen = max(turns_text_length)

        batch_turns_image[i] = turns_image
        batch_turns_text_embedding[i] = turns_text_embedding
        batch_turns_word_embeddings[i, :, :tlen] = \
            turns_word_embeddings
        batch_turns_text_length[i] = turns_text_length

        batch_scene_id.append(b["scene_id"])
        batch_turns_utterence.append(b["turns_utterence"])

    sample = {
        "scene_id": np.array(batch_scene_id),
        "dialogs": np.array(batch_turns_utterence, dtype=np.object),
        "background": torch.FloatTensor(background),
        "turns_image": torch.FloatTensor(batch_turns_image),
        "turns_text_embedding": torch.FloatTensor(batch_turns_text_embedding),
        "turns_word_embeddings": torch.FloatTensor(batch_turns_word_embeddings),
        "turns_text_length": torch.LongTensor(batch_turns_text_length),
        "dialog_length": torch.LongTensor(np.array(dialog_lengths)),
    }
    return sample
