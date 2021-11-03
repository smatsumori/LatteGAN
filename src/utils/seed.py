import os
import random

import numpy as np
import torch


def seed_everything(seed=42):
    """fix seed for pytorch framework

    Parameters
    ----------
    seed : int, optional
        seed value, by default 42
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
