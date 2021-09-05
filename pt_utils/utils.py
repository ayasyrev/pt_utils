import importlib
import os
import random
from dataclasses import asdict
from typing import Any, Union

import numpy as np
import torch
from omegaconf.omegaconf import DictConfig, OmegaConf
from rich.progress import Progress


def flat_dict(cfg_dict: Union[dict, DictConfig]):
    if type(cfg_dict) is not dict:
        if type(cfg_dict) is DictConfig:
            cfg_dict = OmegaConf.to_container(cfg_dict)
        else:  # dataclass like
            cfg_dict = asdict(cfg_dict)
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


def format_time(seconds: float, long: bool = True) -> str:
    "Format secons to mm:ss, optoinal mm:ss.ms"
    seconds_int = int(seconds)
    min, sec = (seconds_int // 60) % 60, seconds_int % 60
    res = f'{min:02d}:{sec:02d}'
    if long:
        res = '.'.join([res, f'{int((seconds - seconds_int) * 10)}'])
    return res


def format_log(to_log: dict) -> str:
    items = []
    for key, value in to_log.items():
        if type(value) is int:
            items.append(str(value))
        elif type(value) is float:
            if 'time' in key:
                items.append(format_time(value))
            else:
                items.append(f"{value:0.4f}")
    return ' '.join([f"{item:>9}" for item in items])


def set_seed(SEED_NUM: int = 2021,
             seed_pythonhash: bool = True,
             seed_random: bool = True,
             seed_numpy: bool = True,
             seed_torch: bool = True,
             torch_benchmark: bool = True,
             torch_deterministic: bool = False,
             ) -> None:
    """Set seeds.
        TODO: check https://pytorch.org/docs/stable/notes/randomness.html?highlight=deterministic
    """
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


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
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
    """Load model state from given path.

        Args:
            model (torch.nn.Module): model for load state
            state_path (str): path for state dictionary.
    """
    state_dict = torch.load(state_path)
    model.load_state_dict(state_dict)


class ProgressBar(Progress):
    '''Progress bar from rich adapted for Learner'''
    def __init__(self, ) -> None:
        super().__init__(transient=True)

    def fit_start(self, epochs, train_dl_len, val_dl_len) -> None:
        self.main_job = self.add_task('fit', total=epochs)
        self.len_train_dl = train_dl_len
        self.len_val_dl = val_dl_len
        self.epochs = epochs
        self.start()

    def fit_end(self):
        self.stop()

    def epoch_start(self, epoch_num):
        self._tasks[self.main_job].description = f"ep {epoch_num} of {self.epochs}"

    def epoch_end(self):
        self.update(self.main_job, advance=1)
        self.remove_task(self.train_job)
        self.remove_task(self.val_job)

    def train_start(self):
        self.train_job = self.add_task('train', total=self.len_train_dl)

    def train_end(self):
        pass

    def train_batch_start(self, batch_num):
        pass

    def train_batch_end(self, batch_num: int = None):
        self.update(self.train_job, advance=1)
        if batch_num is not None:
            self._tasks[self.train_job].description = f"batch {batch_num + 1}/{self.len_train_dl}"

    def val_start(self):
        self.val_job = self.add_task('validate', total=self.len_val_dl)

    def val_batch_end(self):
        self.update(self.val_job, advance=1)

    def val_end(self):
        pass
