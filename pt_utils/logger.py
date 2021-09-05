
from dataclasses import dataclass
from typing import Union
from pathlib import Path, PosixPath
import os

import wandb

from .utils import flat_dict


@dataclass
class LoggerCfg:
    project: str = ''


class Logger:
    # def __init__(self) -> None:

    def start(self, *args, **kwargs):
        pass

    def log(self, metrics: dict):
        pass

    def trace_model(self, model):
        pass

    def log_cfg(self, cfg: dict):
        pass

    def finish(self):
        pass


class LocalLogger(Logger):
    '''Log locally.'''
    def __init__(self,
                 log_path: Union[str, PosixPath] = '.',
                 log_file: str = 'log.csv',
                 cfg_file: str = 'log.cfg',
                 add_data: bool = False) -> None:
        super().__init__()
        self.log_path = Path(log_path)
        self.log_path.mkdir(exist_ok=True)
        self.log_file = log_file
        self.cfg_file = cfg_file
        self.file = None
        self.mode = 'a' if add_data else 'w'

    def start(self, *args, **kwargs):
        self.file = open(self.log_path / self.log_file, mode=self.mode)
        header = kwargs.get('header', None)
        if header is not None:
            self.file.write(','.join(header) + '\n')

    def finish(self):
        self.file.close()

    def log(self, metrics: dict):
        self.file.write(",".join([str(values) for values in metrics.values()]) + '\n')
        self.file.flush()
        os.fsync(self.file.fileno())

    def log_cfg(self, cfg: dict):
        with open(self.log_path / self.cfg_file, 'w') as f:
            for k, v in flat_dict(cfg).items():
                f.write(f"{k}: {v}" + '\n')


@dataclass
class WandbCfg(LoggerCfg):
    log_type: str = 'gradients'  # 'all' / 'parameters' / None


class WandbLogger(Logger):
    def __init__(self, project: Union[str, None] = None, cfg: WandbCfg = WandbCfg()) -> None:
        super().__init__()
        self.project = cfg.project if project is None else project
        self.cfg = cfg

    def start(self, *args, **kwargs):
        self.run = wandb.init(project=self.project)
        self.run.name = '-'.join([self.run.name.split('-')[-1], self.run.id])
        print(f"logger name: {self.run.name}, num: {self.run.name.split('-')[0]}")

    def log(self, metrics: dict):
        self.run.log(metrics)

    def trace_model(self, model):
        self.run.watch(model, log=self.cfg.log_type)

    def log_cfg(self, cfg):
        self.run.config.update(flat_dict(cfg))

    def finish(self):
        self.run.finish()
