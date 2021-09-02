from dataclasses import dataclass, field
import time
from typing import Union

import torch
import torch.nn as nn

from rich import print
from rich.progress import Progress

from accelerate import Accelerator

from pt_utils.logger import LoggerCfg, Logger


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


@dataclass
class LearnerCfg:
    project_name: str = 'test'
    lr: float = 0.001
    # trace_model: bool = False  # add to learner code for trace model
    # logger_cfg: LoggerCfg = field(default_factory=LoggerCfg)


class Learner:
    """Basic wrapper over base train loop.
    Handle model, dataloaders, optimizer and loss function.
    Uses acceleartor as handler over different devices, progress bar and simple logger capabilites."""
    def __init__(self, model: nn.Module, loss_fn, opt_fn, train_dl, val_dl,
                 cfg: LearnerCfg = LearnerCfg(),
                 accelerator: Union[Accelerator, None] = None,
                 batch_tfm: Union[nn.Module, None] = None,
                 logger: Logger = None) -> None:
        if accelerator is None:
            self.accelerator = Accelerator()
        else:
            self.accelerator = accelerator

        self.model = model
        self.loss_fn = loss_fn
        self.opt_fn = opt_fn
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.cfg = cfg

        self.opt = self.reset_opt()

        self.batch_tfm = batch_tfm
        if logger is None:
            logger = Logger()
        self.logger = logger

    def reset_opt(self):
        return self.opt_fn(self.model.parameters(), lr=self.cfg.lr)

    def fit(self, epochs: int):
        self.befor_fit()

        main_job = self.progress_bar.add_task('fit....', total=epochs)
        for epoch in range(1, epochs + 1):
            self.progress_bar.tasks[main_job].description = f"ep {epoch} of {epochs}"
            self.model.train()
            start_time = time.time()
            len_train_dl = len(self.train_dl)
            train_job = self.progress_bar.add_task('train', total=len_train_dl)
            for batch_num, batch in enumerate(self.train_dl):
                self.progress_bar._tasks[train_job].description = f"batch {batch_num}/{len_train_dl}"
                loss = self.loss_batch(batch)
                self.accelerator.backward(loss)
                self.opt.step()
                self.opt.zero_grad()
                self.progress_bar.update(train_job, advance=1)
            train_time = time.time() - start_time
            self.model.eval()
            with torch.no_grad():
                valid_losses = []
                len_val_dl = len(self.val_dl)
                val_job = self.progress_bar.add_task('validate...', total=len_val_dl)
                for batch_num, batch in enumerate(self.val_dl):
                    valid_losses.append(self.loss_batch(batch).item())  # cpu? dont collect -> just summ?
                    self.progress_bar.update(val_job, advance=1)
                valid_loss = sum(valid_losses) / len(valid_losses)
            epoch_time = time.time() - start_time
            val_time = epoch_time - train_time
            to_log = {'epoch': epoch, 'train_loss': loss.item(), 'val_loss': valid_loss,
                      'time': epoch_time, 'train_time': train_time, 'val_time': val_time}
            print(format_log(to_log))
            self.logger.log(to_log)
            self.progress_bar.remove_task(train_job)
            self.progress_bar.remove_task(val_job)
            self.progress_bar.update(main_job, advance=1)
        self.after_fit()

    def loss_batch(self, batch):
        input = batch[0]
        if self.batch_tfm is not None:
            input = self.batch_tfm(input)
        pred = self.model(input)
        return self.loss_fn(pred, batch[1])

    def befor_fit(self):
        header = ['epoch', 'train_loss', 'val_loss', 'time', 'train_time', 'val_time']
        self.train_start_time = time.time()
        self.logger.start(header=header)
        self.logger.log_cfg(self.cfg)
        self.model, self.opt, self.train_dl, self.val_dl = self.accelerator.prepare(self.model, self.opt,
                                                                                    self.train_dl, self.val_dl)
        if self.batch_tfm:
            self.batch_tfm = self.accelerator.prepare(self.batch_tfm)
        self.progress_bar = Progress(transient=True)
        self.progress_bar.start()
        print(' '.join([f"{value:^9}" for value in header]))

    def after_fit(self):
        full_time = time.time() - self.train_start_time
        self.progress_bar.print(f"full time: {format_time(full_time)}")
        self.logger.finish()
        self.progress_bar.stop()
