import h5py
import cv2
import json
import numpy as np

import torch
from torch.utils.data import Dataset


class ICLEVRDataset(Dataset):
    def __init__(self, dataset_path, glove_path):
        super().__init__()
        self.dataset = None

        self.dataset_path = dataset_path

        self.dataset_size = 0
        self.keys = []
        self.background = None

        with h5py.File(dataset_path, "r") as f:
            self.keys = [key for key in f.keys() if key.isdigit()]
            self.dataset_size = len(self.keys)

            background = f["background"][...]
            background = cv2.resize(background, (128, 128))
            background = background.transpose(2, 0, 1)
            self.background = background / 128. - 1

        # build glove vocabs
        self.glove = _parse_glove(glove_path)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.dataset_path, "r")

        index = self.keys[idx]
        example = self.dataset[index]

        images = example["images"][...]
        utterences = example["text"][...]
        objects = example["objects"][...]
        scene_id = example["scene_id"][...]

        utterences = json.loads(eval(str(utterences)))
        utterences_tokenized = [utter.split() for utter in utterences]
        turns_text_length = [len(t) for t in utterences_tokenized]

        max_text_length = max(turns_text_length)
        word_embeddings = np.zeros((len(utterences), max_text_length, 300))
        for i, utter in enumerate(utterences_tokenized):
            for j, w in enumerate(utter):
                word_embeddings[i, j] = self.glove[w]

        images = images[..., ::-1]
        images = images / 128. - 1
        images = images.transpose(0, 3, 1, 2)

        scene_id = eval(str(scene_id)).decode()

        sample = {
            "background": self.background,
            "turns_image": images,
            "turns_word_embeddings": word_embeddings,
            "turns_text_length": turns_text_length,
            "turns_objects": objects,
            "scene_id": scene_id,
            "turns_utterence": utterences,
        }

        return sample


def _parse_glove(glove_path):
    """Construct Dict[word, glove_embedding(300)] from .txt

    Parameters
    ----------
    glove_path : str
        path to glove (word, values) text file.
        e.g.)
        green 0.01 -6.44 1.23 ...
        add 1.11 2.30 -0.13 ...

    Returns
    -------
    glove: dict
        key = word token
        value = 300-dim glove word embedding
    """
    glove = {}
    with open(glove_path, "r") as f:
        for line in f:
            splitline = line.split()
            word = splitline[0]
            embedding = np.array([float(val) for val in splitline[1:]])
            glove[word] = embedding
    return glove


def iclevr_collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x["turns_image"]), reverse=True)

    dialog_lengths = list(map(lambda x: len(x["turns_image"]), batch))
    max_dialog_length = max(dialog_lengths)

    batch_max_text_length = [max(b["turns_text_length"]) for b in batch]
    max_text_length = max(batch_max_text_length)

    batch_size = len(batch)
    _, c, h, w = batch[0]["turns_image"].shape

    stacked_images = np.zeros(
        (batch_size, max_dialog_length, c, h, w))
    stacked_word_embeddings = np.zeros(
        (batch_size, max_dialog_length, max_text_length, 300))
    stacked_text_lengths = np.zeros(
        (batch_size, max_dialog_length))
    stacked_objects = np.zeros(
        (batch_size, max_dialog_length, 24))

    scene_ids = []
    dialogs = []

    background = None
    for i, b in enumerate(batch):
        background = b["background"]

        turns_image = b["turns_image"]
        turns_word_embeddings = b["turns_word_embeddings"]
        turns_text_length = b["turns_text_length"]
        turns_objects = b["turns_objects"]

        dialog_length = turns_image.shape[0]

        stacked_images[i, :dialog_length] = \
            turns_image
        stacked_word_embeddings[i, :dialog_length, :max(turns_text_length)] = \
            turns_word_embeddings
        stacked_text_lengths[i, :dialog_length] = \
            np.array(turns_text_length)
        stacked_objects[i, :dialog_length] = \
            turns_objects

        scene_ids.append(b["scene_id"])
        dialogs.append(b["turns_utterence"])

    # BUG: it causes to add noise to GT images of valid and test.
    # stacked_images += np.random.uniform(
    #     size=stacked_images.shape, low=0., high=1. / 64)

    sample = {
        "scene_id": np.array(scene_ids),
        "dialogs": np.array(dialogs, dtype=object),
        "background": torch.FloatTensor(background),
        "turns_image": torch.FloatTensor(stacked_images),
        "turns_word_embeddings": torch.FloatTensor(stacked_word_embeddings),
        "turns_text_length": torch.LongTensor(stacked_text_lengths),
        "turns_objects": torch.FloatTensor(stacked_objects),
        "dialog_length": torch.LongTensor(np.array(dialog_lengths)),
    }

    return sample
