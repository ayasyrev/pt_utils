
from typing import Union
from pathlib import Path, PosixPath
import os

from torch.utils.tensorboard import SummaryWriter
import wandb
from omegaconf import OmegaConf

from .utils import flat_dict


class Logger:
    # def __init__(self) -> None:

    def start(self, *args, **kwargs):
        pass

    def log(self, metrics: dict):
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
                 add_data: bool = False,
                 project: str = '') -> None:
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
            # for k, v in flat_dict(cfg).items():
            #     f.write(f"{k}: {v}" + '\n')
            f.write(OmegaConf.to_yaml(cfg, resolve=True))


class WandbLogger(Logger):
    def __init__(self, project: Union[str, None] = None,
                 trace_model: bool = False,
                 log_type: str = None  # 'gradients' / 'all' / 'parameters' / None
                 ) -> None:
        super().__init__()
        self.project = 'def_name' if project is None else project
        self.log_type = log_type
        self.trace_model = trace_model

    def start(self, model=None, *args, **kwargs):
        self.run = wandb.init(project=self.project)
        self.run.name = '-'.join([self.run.name.split('-')[-1], self.run.id])
        print(f"logger name: {self.run.name}, num: {self.run.name.split('-')[0]}")
        if self.trace_model:
            if model is not None:
                self.run.watch(model, log=self.log_type)

    def log(self, metrics: dict):
        self.run.log(metrics)

    def log_cfg(self, cfg):
        self.run.config.update(flat_dict(cfg))

    def finish(self):
        self.run.finish()


class TensorBoardLogger(Logger):
    def __init__(self, log_dir: Union[str, None] = None) -> None:
        super().__init__()
        self.log_dir = log_dir or 'def_name'

    def start(self, *args, **kwargs):
        self.writer = SummaryWriter(log_dir=self.log_dir)

    # def log_cfg(self, cfg: dict):
    #     self.hparams = clear_dict(flat_dict(cfg))

    def log(self, metrics: dict):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, global_step=metrics['epoch'])
        self.last_log = metrics

    def finish(self):
        # self.writer.add_hparams(self.hparams, {'acc': self.last_log['accuracy']})
        self.writer.close()
