import h5py
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


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


class CoDrawPropV1TrainDataset(Dataset):
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
        self.keys = []

        with h5py.File(dataset_path, "r") as f:
            self.background = f["background"][...]

            # keys = [(dialog_index, turn_index), ...]
            _keys = [key for key in f.keys() if key.isdigit()]
            for key in _keys:
                dialog_length = f[key]["objects"][...].shape[0]
                self.keys.extend([(key, t) for t in range(dialog_length)])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.dataset_path, "r")
        if self.embed_dataset is None:
            self.embed_dataset = h5py.File(self.embed_dataset_path, "r")

        dialog_index, turn_index = self.keys[idx]

        # fetch data from original dataset
        example = self.dataset[dialog_index]
        image = example["images"][...][turn_index]
        utter = example["utterences"][...][turn_index]
        objects = example["objects"][...][turn_index]
        scene_id = example["scene_id"][...]
        if turn_index > 0:
            prev_image = example["images"][...][turn_index - 1]
            prev_objects = example["objects"][...][turn_index - 1]
        else:
            prev_image = self.background
            prev_objects = np.zeros_like(objects)

        # fetch embeddings of utter from generated dataset
        embed_example = self.embed_dataset[dialog_index]
        text_embedding = embed_example["turns_text_embedding"][...][turn_index]
        word_embeddings = embed_example["turns_word_embeddings"][...][turn_index]
        text_length = embed_example["turns_text_length"][...][turn_index]

        # image preprocessing
        image = _image_preprocessing(image, self.image_size, self.image_size)
        prev_image = _image_preprocessing(
            prev_image, self.image_size, self.image_size)

        # text preprocessing
        utter = utter.decode()
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


def codraw_propv1_train_collate_fn(batch):
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
        (batch_size, 58), dtype=np.float32)
    batch_added_objects = np.zeros(
        (batch_size, 58), dtype=np.float32)
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


class CoDrawPropV1EvalDataset:
    def __init__(
        self,
        dataset_path,
        embed_dataset_path,
        batch_size,
        image_size=128,
        **kwargs,
    ):
        """__init__.

        Parameters
        ----------
        dataset_path :
            dataset_path
        embed_dataset_path :
            An h5 path to precomputed text embeddings.
        batch_size :
            batch_size
        image_size :
            image_size
        kwargs :
            kwargs
        """
        self.dataset_path = dataset_path
        self.embed_dataset_path = embed_dataset_path
        self.batch_size = batch_size
        self.image_size = image_size

        self.dataset_size = 0
        self.dataset = None
        self.embed_dataset = None
        self.keys = []
        self.background = None  # a placeholder for a bg image

        with h5py.File(dataset_path, "r") as f:
            # f.keys() = dataset_size + 1(background)
            self.dataset_size = len(list(f.keys())) - 1

            dialog_lengths = []
            for i in range(self.dataset_size):
                # objects.shape: (n_turn, 58)
                dialog_lengths.append(f[str(i)]["objects"].shape[0])
            # sort data index by decending order of dialog length
            self.keys = np.argsort(np.array(dialog_lengths))[::-1]

            background = f["background"][...]
            background = background[..., ::-1].transpose(2, 0, 1)
            self.background = background / 128. - 1

        # chunking indices by similar dialog length
        self.blocks_maps = {}
        for i in range(0, self.dataset_size, batch_size):
            block_key = i // batch_size
            self.blocks_maps[block_key] = self.keys[i:i + batch_size]
        self.blocks_keys = np.array(list(self.blocks_maps.keys()))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.dataset_path, "r")
        if self.embed_dataset is None:
            self.embed_dataset = h5py.File(self.embed_dataset_path, "r")

        block_index = self.blocks_keys[idx // self.batch_size]
        sample_index = idx % self.batch_size

        if sample_index > len(self.blocks_maps[block_index]) - 1:
            sample_index = len(self.blocks_maps[block_index]) - 1

        # get the original index
        index = self.blocks_maps[block_index][sample_index]

        # TODO: check if both indices correspond to correct scenes
        # fetch data from original dataset
        example1 = self.dataset[str(index)]
        images = example1["images"][...]  # (L, H, W, C)
        utterences = example1["utterences"][...]  # (L,)
        scene_id = example1["scene_id"][...]

        # fetch embeddings of utter from generated dataset
        example2 = self.embed_dataset[str(index)]
        turns_text_embedding = \
            example2["turns_text_embedding"][...]  # (L, D)
        turns_word_embeddings = \
            example2["turns_word_embeddings"][...]  # (L, S, D)
        turns_text_length = \
            example2["turns_text_length"][...]  # (L,)

        # image preprocessing
        images = _image_preprocessing(images, self.image_size, self.image_size)

        # text preprocessing
        utterences = [uttr.decode() for uttr in utterences]

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


def codraw_propv1_eval_collate_fn(batch):
    # sort samples in batch by descending order of dialog length
    batch = sorted(batch, key=lambda x: len(x["turns_image"]), reverse=True)

    dialog_lengths = list(map(lambda x: len(x["turns_image"]), batch))
    max_dialog_length = max(dialog_lengths)

    batch_max_text_length = [max(b["turns_text_length"]) for b in batch]
    max_text_length = max(batch_max_text_length)

    batch_size = len(batch)
    _, c, h, w = batch[0]["turns_image"].shape
    _, d = batch[0]["turns_text_embedding"].shape

    # placeholders
    batch_turns_image = np.zeros(
        (batch_size, max_dialog_length, c, h, w),
        dtype=np.float32,
    )
    batch_turns_text_embedding = np.zeros(
        (batch_size, max_dialog_length, d),
        dtype=np.float32,
    )
    batch_turns_word_embeddings = np.zeros(
        (batch_size, max_dialog_length, max_text_length, d),
        dtype=np.float32,
    )
    batch_turns_text_length = np.zeros(
        (batch_size, max_dialog_length),
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

        dlen = turns_image.shape[0]
        tlen = max(turns_text_length)

        batch_turns_image[i, :dlen] = \
            turns_image
        batch_turns_text_embedding[i, :dlen] = \
            turns_text_embedding
        batch_turns_word_embeddings[i, :dlen, :tlen] = \
            turns_word_embeddings
        batch_turns_text_length[i, :dlen] = \
            turns_text_length

        batch_scene_id.append(b["scene_id"])
        batch_turns_utterence.append(b["turns_utterence"])

    # BUG: it causes to add noise to GT images of valid and test.
    # stacked_images += np.random.uniform(
    #     size=stacked_images.shape, low=0., high=1. / 64)

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
