import os
import random

import numpy as np
import torch


def flat_dict(cfg_dict: dict):
    res = {}
    for item, value in cfg_dict.items():
        res.update(_unfold(item, value))
    return res


def _unfold(parent, element):
    res = {}
    if isinstance(element, dict):
        for item, value in element.items():
            res.update(_unfold(f"{parent}.{item}", value))
    else:
        res[parent] = element
    return res


def set_seed(SEED_NUM: int = 2021,
             seed_pythonhash: bool = True,
             seed_random: bool = True,
             seed_numpy: bool = True,
             seed_torch: bool = True,
             torch_benchmark: bool = True,
             torch_deterministic: bool = False,
             ) -> None:
    print(f"seeding: {SEED_NUM}")
    if seed_pythonhash:
        os.environ['PYTHONHASHSEED'] = str(SEED_NUM)
    if seed_random:
        random.seed(SEED_NUM)
    if seed_numpy:
        np.random.seed(SEED_NUM)
    if seed_torch:
        torch.manual_seed(SEED_NUM)
        torch.cuda.manual_seed(SEED_NUM)
        torch.cuda.manual_seed_all(SEED_NUM)

    torch.backends.cudnn.benchmark = torch_benchmark
    torch.backends.cudnn.deterministic = torch_deterministic
