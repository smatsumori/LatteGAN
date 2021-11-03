import os

import torch

from logging import getLogger
logger = getLogger(__name__)


def set_visible_gpus(gpu_ids="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids


def show_gpu_info():
    logger.info("current device: {}".format(torch.cuda.current_device()))
    avail = torch.cuda.is_available()
    logger.info("torch cuda is available: {}".format(avail))
    if avail:
        cnt = torch.cuda.device_count()
        logger.info("device count: {}".format(cnt))
        for i in range(cnt):
            logger.info("[{}] device: {} device name: {}".format(
                i,
                torch.cuda.device(i),
                torch.cuda.get_device_name(i),
            ))
