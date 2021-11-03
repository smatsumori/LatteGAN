import os
import re
from ast import literal_eval

import h5py
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets.folder import has_file_allowed_extension

from modules.metrics.object_localizer import Inception3ObjectLocalizer

from logging import getLogger
logger = getLogger(__name__)


loaded_model = None
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


class ImageFolderNonGT(Dataset):
    def __init__(self, root, dataset_path, transform):
        """__init__.

        Parameters
        ----------
        root :
            A path to gt and generated images e.g. ./iamges_valid
        dataset_path :
            A path to h5 dataset.
        transform :
        """
        self.root = root
        self.dataset_path = dataset_path
        self.transform = transform

        # directory name of generated images == scene_id
        gen_dirs = []
        for d in os.scandir(self.root):
            if d.is_dir() and not d.name.endswith("_gt"):
                # dirs with generated images should not contain `_gt`
                gen_dirs.append(d.name)

        gt_files, gen_files = [], []
        final_timestep_gen_files = []
        for dname in gen_dirs:
            gt_files.extend(self._get_all_image_path(dname + "_gt")[0])
            _gen_files, _final_gen_file = self._get_all_image_path(dname)
            gen_files.extend(_gen_files)
            final_timestep_gen_files.append(_final_gen_file)

        self.gt_files = gt_files
        self.gen_files = gen_files
        self.final_timestep_gen_files = final_timestep_gen_files

        # load original dataset ... varidate max score
        self.dataset = None
        self.id2idx = {}
        with h5py.File(dataset_path, "r") as f:
            # _keys = [access_id, ...]
            # NOTE: access_id != scene_id, access_id is an access key of h5 to each episodes.
            # NOTE: f.keys() = [{access_id}, ..., background, entities]
            _keys = [key for key in f.keys() if key.isdigit()]
            for k in _keys:
                example = f[k]
                scene_id = str(example["scene_id"][...])
                # if iclevr, scene_id must be decode as str
                # "entities" is in only iclevr
                if "entities" in f.keys():
                    scene_id = literal_eval(scene_id).decode()
                self.id2idx[scene_id] = k
                assert scene_id in gen_dirs

    def _get_all_image_path(self, dir_name):
        """_get_all_image_path.

        Parameters
        ----------
        dir_name :
            {scene_id}[_gt]
        """
        files = os.listdir(os.path.join(self.root, dir_name))
        files = [
            x for x in files
            if has_file_allowed_extension(x, IMG_EXTENSIONS)
        ]
        files = sorted(files, key=natural_keys)
        files = [os.path.join(self.root, dir_name, f) for f in files]
        return files, files[-1]

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.dataset_path, "r")

        # load from results directory
        gt_path = self.gt_files[idx]
        gen_path = self.gen_files[idx]
        # check both consist the same turn
        assert os.path.basename(gt_path) == os.path.basename(gen_path)
        gt_sample = cv2.imread(gt_path)[..., ::-1]
        gen_sample = cv2.imread(gen_path)[..., ::-1]

        # load from h5 dataset
        scene_id = gen_path.split("/")[-2]
        i = self.id2idx[scene_id]
        j = int(gen_path.split("/")[-1].split(".")[0])
        example = self.dataset[i]
        objects = example["objects"][...][j]

        # varidate gt image is correct
        _tmp_sample = example["images"][...][j]
        _tmp_sample = _tmp_sample[..., ::-1]
        if not np.all(gt_sample == _tmp_sample):
            logger.warning(f"GT image mismatch: {gt_path}")

        # flag if final timestep sample
        is_final_timestep = gen_path in self.final_timestep_gen_files

        # to tensor
        gt_sample = self.transform(gt_sample)
        gen_sample = self.transform(gen_sample)
        objects = torch.from_numpy(objects)
        is_final_timestep = torch.tensor(is_final_timestep)

        return gen_sample, gt_sample, objects, is_final_timestep


def setup_inception_model(num_classes, pretrained=False):
    if num_classes == 24:
        num_coords = 2
    else:
        num_coords = 3
    model = nn.DataParallel(Inception3ObjectLocalizer(
        num_objects=num_classes,
        pretrained=pretrained,
        num_coords=num_coords)).cuda()
    return model


def _init_inception(model_path):
    global loaded_model

    checkpoint = torch.load(model_path)
    if checkpoint['cuda_enabled']:
        torch.backends.cudnn.deterministic = True
    loaded_model = setup_inception_model(
        checkpoint['num_classes'], pretrained=checkpoint['pretrained']
    )
    if checkpoint['cuda_enabled']:
        loaded_model = loaded_model.cuda()
        # see t.ly/SWK2 for more information.
        # torch.backends.cudnn.benchmark = True
    loaded_model.load_state_dict(checkpoint['state_dict'])
    loaded_model.eval()


def construct_graph(coords):
    n = len(coords)
    graph = np.zeros((2, n, n))
    for i in range(n):
        if coords.shape[1] == 2:
            ref_x, ref_y = coords[i]
        else:
            ref_x, _, ref_y = coords[i]
        for j in range(n):
            if i == j:
                query_x, query_y = 0.5, 0.5
            else:
                if coords.shape[1] == 2:
                    query_x, query_y = coords[j]
                else:
                    query_x, _, query_y = coords[j]

            if ref_x > query_x:
                graph[0, i, j] = 1
            elif ref_x < query_x:
                graph[0, i, j] = -1

            if ref_y > query_y:
                graph[1, i, j] = 1
            elif ref_y < query_y:
                graph[1, i, j] = -1

    return graph


def get_graph_similarity(detections, label, locations, gt_locations):
    """Computes the accuracy of relationships of the intersected
    detections multiplied by recall
    """
    intersection = (detections & label).astype(bool)
    if not np.any(intersection):
        return 0

    locations = locations.data.cpu().numpy()[intersection]
    gt_locations = gt_locations.data.cpu().numpy()[intersection]

    genereated_graph = construct_graph(locations)
    gt_graph = construct_graph(gt_locations)

    matches = (genereated_graph == gt_graph).astype(int).flatten()
    matches_accuracy = matches.sum() / len(matches)
    recall = recall_score(label, detections, average='samples')

    graph_similarity = recall * matches_accuracy

    return graph_similarity


def get_obj_det_acc(dataloader):
    """Returns object detection accuracy.

    Parameters
    ----------
    dataloader :
        A dataloader instance.
    """
    graph_similarity = []

    pred_all = []
    gt_all = []
    objs_all = []

    for _, (sample, gt, objs, flag) in enumerate(tqdm(dataloader)):
        sample = sample.cuda()
        gt = gt.cuda()

        detection_logits, locations = loaded_model(sample)
        gt_detection_logits, gt_locations = loaded_model(gt)

        pred = detection_logits > 0.5
        gt_pred = gt_detection_logits > 0.5
        objs = objs > 0.5

        pred = pred.cpu().numpy().astype(np.int64)
        gt_pred = gt_pred.cpu().numpy().astype(np.int64)
        objs = objs.cpu().numpy().astype(np.int64)

        pred_all.append(pred)
        gt_all.append(gt_pred)
        objs_all.append(objs)

        for i in range(sample.size(0)):
            if flag[i]:
                graph_similarity.append(
                    get_graph_similarity(
                        pred[i][None, ...],
                        gt_pred[i][None, ...],
                        locations[i].unsqueeze(dim=0),
                        gt_locations[i].unsqueeze(dim=0),
                    )
                )

    pred_all = np.concatenate(pred_all, axis=0)
    gt_all = np.concatenate(gt_all, axis=0)
    objs_all = np.concatenate(objs_all, axis=0)

    # pred_all.shape: (n_samples, n_classes) (True or False)
    # use multilabel confusion matrix
    # cmat.shape: (n_classes, 2, 2)
    cmat = multilabel_confusion_matrix(
        gt_all,
        pred_all,
        labels=np.arange(gt_all.shape[1]),
    )
    ps = precision_score(gt_all, pred_all, average="samples", zero_division=0)
    rs = recall_score(gt_all, pred_all, average="samples", zero_division=0)
    f1 = f1_score(gt_all, pred_all, average="samples", zero_division=0)

    # reference score
    print("Reference AP: {:.5f}".format(
        precision_score(objs_all, gt_all, average="samples", zero_division=0)))
    print("Reference AR: {:.5f}".format(
        recall_score(objs_all, gt_all, average="samples", zero_division=0)))
    print("Reference F1: {:.5f}".format(
        f1_score(objs_all, gt_all, average="samples", zero_division=0)))

    return ps, rs, f1, graph_similarity, cmat


def calculate_inception_objects_accuracy(
    image_dir,
    model_path,
    dataset_path,
    batch_size=1,
    num_workers=0,
):
    """Caluculate AP, AR, F1, RSIM score by GeNeVA author provided method.

    Args:
        image_dir (str): path to directory, submission format described README.
        model_path (str): path to pretrained object detector and
            localizer weights.

    Returns:
        tuple of int: [AP, AR, F1, CosineSim, RSIM, Confusion Matrix]
    """
    if loaded_model is None:
        _init_inception(model_path)

    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.ToTensor(),
    ])
    dataset = ImageFolderNonGT(
        image_dir,
        dataset_path,
        transform=test_transforms,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=False,
    )

    with torch.no_grad():
        avg_precision, avg_recall, avg_f1, graph_similarity, cmat = \
            get_obj_det_acc(dataloader)
        gsim = np.mean(graph_similarity)

        # AP, AR, F1
        print("\nNumber of images used: {}\n AP: {}\n AR: {}\n F1: {}".format(
            len(dataset), avg_precision, avg_recall, avg_f1,
        ))

        # RSIM
        print("\nNumber of images used: {}\n GS: {}".format(
            len(graph_similarity), gsim
        ))

    return avg_precision, avg_recall, avg_f1, None, gsim, cmat
