
from dataclasses import dataclass, asdict
from typing import Union

import wandb

from .utils import flat_dict


class Logger:
    def __init__(self, project: str = '', cfg: dict = None) -> None:
        self.project = project  # todo remove from default
        self.cfg = cfg

    def start(self, *args, **kwargs):
        pass

    def log(self, metrics):
        pass

    def trace_model(self, model):
        pass

    def log_cfg(self, cfg):
        pass

    def finish(self):
        pass


@dataclass
class LoggerCfg:
    project: str = ''


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
