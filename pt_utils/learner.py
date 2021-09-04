import time
from typing import Union

import torch
import torch.nn as nn

from rich import print

from accelerate import Accelerator

from pt_utils.logger import Logger
from pt_utils.utils import format_log, format_time, ProgressBar


class Learner:
    """Basic wrapper over base train loop.
    Handle model, dataloaders, optimizer and loss function.
    Uses acceleartor as handler over different devices, progress bar and simple logger capabilites."""
    def __init__(self, model: nn.Module,
                 train_dl, val_dl,
                 opt_fn, loss_fn,
                 accelerator: Union[Accelerator, None] = None,
                 batch_tfm: Union[nn.Module, None] = None,
                 logger: Logger = None,
                 cfg: dict = None) -> None:
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
        self.befor_fit(epochs)

        for epoch in range(1, epochs + 1):
            self.progress_bar.epoch_start(epoch)
            self.model.train()
            start_time = time.time()

            # train
            self.progress_bar.train_start()
            for batch_num, batch in enumerate(self.train_dl):
                loss = self.loss_batch(batch)
                self.accelerator.backward(loss)
                self.opt.step()
                self.opt.zero_grad()
                self.progress_bar.train_batch_end(batch_num)
            train_time = time.time() - start_time

            # validate
            self.progress_bar.val_start()
            self.model.eval()
            with torch.no_grad():
                valid_losses = []
                for batch_num, batch in enumerate(self.val_dl):
                    valid_losses.append(self.loss_batch(batch).item())  # cpu? dont collect -> just summ?
                    self.progress_bar.val_batch_end()
                valid_loss = sum(valid_losses) / len(valid_losses)
            epoch_time = time.time() - start_time
            val_time = epoch_time - train_time
            to_log = {'epoch': epoch, 'train_loss': loss.item(), 'val_loss': valid_loss,
                      'time': epoch_time, 'train_time': train_time, 'val_time': val_time}
            print(format_log(to_log))
            self.logger.log(to_log)
            self.progress_bar.epoch_end()
        self.after_fit()

    def loss_batch(self, batch):
        input = batch[0]
        if self.batch_tfm is not None:
            input = self.batch_tfm(input)
        pred = self.model(input)
        return self.loss_fn(pred, batch[1])

    def befor_fit(self, epochs):
        header = ['epoch', 'train_loss', 'val_loss', 'time', 'train_time', 'val_time']
        self.train_start_time = time.time()
        self.logger.start(header=header)
        self.logger.log_cfg(self.cfg)
        self.model, self.opt, self.train_dl, self.val_dl = self.accelerator.prepare(self.model, self.opt,
                                                                                    self.train_dl, self.val_dl)
        if self.batch_tfm:
            self.batch_tfm = self.accelerator.prepare(self.batch_tfm)
        self.progress_bar = ProgressBar()
        self.progress_bar.fit_start(epochs, train_dl_len=len(self.train_dl), val_dl_len=len(self.val_dl))

        print(' '.join([f"{value:^9}" for value in header]))

    def after_fit(self):
        full_time = time.time() - self.train_start_time
        print(f"full time: {format_time(full_time)}")
        self.logger.finish()
        self.progress_bar.fit_end()
