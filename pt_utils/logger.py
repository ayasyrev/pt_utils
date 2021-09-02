
from dataclasses import dataclass, asdict
from typing import Union
from pathlib import Path, PosixPath
import os

from omegaconf import DictConfig
import wandb

from .utils import flat_dict


@dataclass
class LoggerCfg:
    project: str = ''


class Logger:
    def __init__(self, project: str = '', cfg: dict = None) -> None:
        self.project = project  # todo remove from default
        self.cfg = cfg

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


@dataclass
class LocalLoggerCfg(LoggerCfg):
    log_path: Union[str, PosixPath] = '.'
    log_file: str = 'log.csv'
    cfg_file: str = 'log.cfg'
    add_data: bool = False


class LocalLogger(Logger):
    '''Log locally.'''
    def __init__(self,
                 project: Union[str, None] = None,
                 cfg: LocalLoggerCfg = None) -> None:
        if cfg is None:
            cfg = LocalLoggerCfg()
        if project is None:
            project = cfg.project
        super().__init__(project=project, cfg=cfg)
        # log_path = '.' if log_path is None else log_path
        self.log_path = Path(self.cfg.log_path)
        self.log_path.mkdir(exist_ok=True)
        # self.log_file = 'log.csv' if log_file is None else log_file
        # self.cfg_file = 'log.cfg' if cfg_file is None else cfg_file
        # self.file = None
        self.mode = 'a' if self.cfg.add_data else 'w'

    def start(self, *args, **kwargs):
        self.file = open(self.log_path / self.cfg.log_file, mode=self.mode)
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
        with open(self.log_path / self.cfg.cfg_file, 'w') as f:
            if type(cfg) is not DictConfig:
                cfg = asdict(cfg)
            for k, v in flat_dict(cfg).items():
                f.write(f"{k}: {v}" + '\n')


@dataclass
class WandbCfg(LoggerCfg):
    log_type: str = 'gradients'  # 'all' / 'parameters' / None


class WandbLogger(Logger):
    def __init__(self, project: Union[str, None] = None, cfg: WandbCfg = WandbCfg()) -> None:
        if project is None:
            project = cfg.project
        super().__init__(project, cfg)

    def start(self, *args, **kwargs):
        self.run = wandb.init(project=self.project)
        self.run.name = '-'.join([self.run.name.split('-')[-1], self.run.id])
        print(f"logger name: {self.run.name}, num: {self.run.name.split('-')[0]}")

    def log(self, metrics: dict):
        self.run.log(metrics)

    # def watch(self, model):
    def trace_model(self, model):
        self.run.watch(model, log=self.cfg.log_type)

    def log_cfg(self, cfg):
        self.run.config.update(flat_dict(asdict(cfg)))

    def finish(self):
        self.run.finish()
