import importlib
import os
import random
from typing import Any

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


def load_obj(obj_path: str = 'xxx.yyy', default_obj_path: str = "") -> Any:
    """Extract an object from a given path.
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.

    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            "Object `{}` cannot be loaded from `{}`.".format(obj_name, obj_path)
        )
    return getattr(module_obj, obj_name)


def load_model_state(model: torch.nn.Module, state_path: str) -> None:
    state_dict = torch.load(state_path)
    model.load_state_dict(state_dict)
