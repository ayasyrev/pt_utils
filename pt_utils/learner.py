from dataclasses import dataclass, field
import time
from typing import Callable, Union
from rich import progress

import torch
import torch.nn as nn

from rich.progress import Progress

from accelerate import Accelerator

from pt_utils.logger import LoggerCfg, Logger


def format_time(seconds: float, long: bool = True):
    "Format secons to mm:ss, optoinal mm:ss.ms"
    seconds_int = int(seconds)
    min, sec = (seconds_int//60)%60, seconds_int%60
    res = f'{min:02d}:{sec:02d}'
    if long:
        res = '.'.join([res, f'{int((seconds - seconds_int) * 10)}'])
    return res


accelerator = Accelerator()
device = accelerator.device


@dataclass
class LearnerCfg:
    project_name: str = 'test'
    lr: float = 0.001
    trace_model: bool = False
    logger_cfg: LoggerCfg = field(default_factory=LoggerCfg)


class Learner:
    """Basic wrapper over base train loop.
    Handle model, dataloaders, optimizer and loss function.
    Uses acceleartor as handler over different devices, progress bar and simple logger capabilites."""
    def __init__(self, model: nn.Module, loss_fn, opt_fn, train_dl, val_dl,
                 cfg: LearnerCfg = LearnerCfg(),
                 device: Accelerator.device = device,
                 batch_tfm: Union[Callable, None] = None,
                 logger: Logger = None) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.opt_fn = opt_fn
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.cfg = cfg

        self.device = device
        self.opt = self.reset_opt()

        self.batch_tfm = batch_tfm
        if logger is None:
            logger = Logger(project=self.cfg.project_name, cfg=self.cfg.logger_cfg)
        self.logger = logger

    def reset_opt(self):
        return self.opt_fn(self.model.parameters(), lr=self.cfg.lr)

    def fit(self, epochs: int):
        train_start_time = time.time()
        self.logger.start()
        self.logger.log_cfg(self.cfg)
        self.model, self.opt, self.train_dl, self.val_dl = accelerator.prepare(self.model, self.opt,
                                                                               self.train_dl, self.val_dl)
        self.progress_bar = Progress()
        with self.progress_bar:
            header = ['epoch', 'train_loss', 'val loss', 'time', 'train time', 'val_time', 'val_time', 'train_time']
            self.progress_bar.print(' '.join([f"{value:>9}" for value in header]))
            main_job = self.progress_bar.add_task('fit....', total=epochs)
            for epoch in range(epochs):
                self.progress_bar.tasks[main_job].description = f"ep {epoch + 1} of {epochs}"
                self.model.train()
                start_time = time.time()
                len_train_dl = len(self.train_dl)
                train_job = self.progress_bar.add_task('train', total=len_train_dl)
                for batch_num, batch in enumerate(self.train_dl):
                    loss = self.loss_batch(batch)
                    self.progress_bar._tasks[train_job].description = f"batch {batch_num}/{len_train_dl}, loss {loss:0.4f}"
                    accelerator.backward(loss)
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
                        valid_losses.append(self.loss_batch(batch).item())  # cpu?
                        self.progress_bar.update(val_job, advance=1)
                    valid_loss = sum(valid_losses) / len(valid_losses)
                epoch_time = time.time() - start_time
                val_time = epoch_time - train_time
                to_log = {'epoch': epoch, 'train_loss': loss, 'val_loss': valid_loss, 'time': epoch_time, 'train_time': train_time, 'val_time': val_time}
                to_progress_bar = [f"{epoch + 1}", f"{loss:0.4f}", f"{valid_loss:0.4f}", f"{format_time(epoch_time)}", f"{format_time(train_time)}", f"{format_time(val_time)}"]
                self.progress_bar.print(' '.join([f"{value:>9}" for value in to_progress_bar]))
                self.logger.log(to_log)
                self.progress_bar.remove_task(train_job)
                self.progress_bar.remove_task(val_job)
                self.progress_bar.update(main_job, advance=1)
        full_time = time.time() - train_start_time
        self.progress_bar.print(f"full time: {format_time(full_time)}")
        self.logger.log({'full_time': full_time})
        self.logger.finish()

    def loss_batch(self, batch):
        input = batch[0]
        if self.batch_tfm is not None:
            input = self.batch_tfm(input)
        pred = self.model(input)
        return self.loss_fn(pred, batch[1])
