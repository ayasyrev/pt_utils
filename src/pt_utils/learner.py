import time
from typing import List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from rich import print

from accelerate import Accelerator

from pt_utils.logger import Logger
from pt_utils.utils import format_log, format_time, ProgressBar
from pt_utils.metrics import accuracy


class Learner:
    """Basic wrapper over base train loop.
    Handle model, dataloaders, optimizer and loss function.
    Uses accelerator as handler over different devices, progress bar and simple logger capabilites."""

    def __init__(
        self,
        model: nn.Module,
        train_dl: Dataloader,
        val_dl: Dataloader,
        opt_fn,
        loss_fn,
        accelerator: Union[Accelerator, None] = None,
        batch_tfm: Union[nn.Module, None] = None,
        logger: Union[Logger, List[Logger], None] = None,
        progress: bool = True,  # use progress
        cfg: Union[dict, None] = None,
    ) -> None:

        if accelerator is None:
            self.accelerator = Accelerator()
        else:
            self.accelerator = accelerator

        self.model: nn.Module = model
        self.loss_fn = loss_fn
        self.opt_fn = opt_fn
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.cfg = cfg

        self.opt = self.reset_opt()

        self.batch_tfm = batch_tfm
        if logger is None:
            self.loggers = [Logger()]
        elif isinstance(logger, list):
            self.loggers = logger
        else:
            self.loggers = [logger]
        self.progress = progress

    def reset_opt(self):
        return self.opt_fn(self.model.parameters(), lr=self.cfg.lr)

    def fit(self, epochs: int):
        self.before_fit(epochs)

        for epoch in range(1, epochs + 1):
            self.epoch_start(epoch)
            self.train()
            self.validate()
            self.epoch_end()
        self.after_fit()

    def one_batch(self, input):
        if self.batch_tfm is not None:
            input = self.batch_tfm(input)
        pred = self.model(input)
        return pred

    def train(self) -> None:
        start_time = time.time()
        self.model.train()
        for batch_num, batch in enumerate(self.train_dl):
            loss = self.loss_fn(self.one_batch(batch[0]), batch[1])
            self.accelerator.backward(loss)
            self.opt.step()
            # self.opt.zero_grad(set_to_none=True)
            for param in self.model.parameters():
                param.grad = None
            if self.progress:
                self.progress_bar.train_batch_end(batch_num)
        self.last_loss = loss.item()
        self.train_time = time.time() - start_time

    def validate(self) -> None:
        start_time = time.time()
        if self.progress:
            self.progress_bar.val_start()
        self.model.eval()
        with torch.no_grad():
            # valid_losses = []
            valid_losses = torch.tensor(0.0, device=self.accelerator.device)
            acc = torch.tensor(0.0, device=self.accelerator.device)
            for batch_num, batch in enumerate(self.val_dl):
                # valid_losses.append(self.loss_batch(batch).item())  # cpu? dont collect -> just summ?
                pred = self.one_batch(batch[0])
                valid_losses.add_(self.loss_fn(pred, batch[1]))
                acc.add_(accuracy(pred, batch[1])[0][0])
                if self.progress:
                    self.progress_bar.val_batch_end()
            # self.valid_loss = sum(valid_losses) / len(valid_losses)
            self.valid_loss = valid_losses.item() / len(self.val_dl)
            self.accuracy = acc.item() / len(self.val_dl)
        self.val_time = time.time() - start_time

    def epoch_start(self, epoch) -> None:
        self.epoch = epoch
        if self.progress:
            self.progress_bar.epoch_start(epoch)
            self.progress_bar.train_start()
        self.epoch_start_time = time.time()

    def epoch_end(self) -> None:
        epoch_time = time.time() - self.epoch_start_time
        to_log = {
            "epoch": self.epoch,
            "train_loss": self.last_loss,
            "val_loss": self.valid_loss,
            "accuracy": self.accuracy,
            "time": epoch_time,
            "train_time": self.train_time,
            "val_time": self.val_time,
        }
        print(format_log(to_log))
        for logger in self.loggers:
            logger.log(to_log)
        if self.progress:
            self.progress_bar.epoch_end()

    def before_fit(self, epochs):
        header = [
            "epoch",
            "train_loss",
            "val_loss",
            "accuracy",
            "time",
            "train_time",
            "val_time",
        ]
        self.train_start_time = time.time()
        self.model, self.opt, self.train_dl, self.val_dl = self.accelerator.prepare(
            self.model, self.opt, self.train_dl, self.val_dl
        )
        for logger in self.loggers:
            logger.start(header=header, model=self.model)
            logger.log_cfg(self.cfg)
        if self.batch_tfm:
            self.batch_tfm = self.accelerator.prepare(self.batch_tfm)
        if self.progress:
            self.progress_bar = ProgressBar()
            self.progress_bar.fit_start(
                epochs, train_dl_len=len(self.train_dl), val_dl_len=len(self.val_dl)
            )

        print(" ".join([f"{value:^9}" for value in header]))

    def after_fit(self):
        full_time = time.time() - self.train_start_time
        print(f"full time: {format_time(full_time)}")
        for logger in self.loggers:
            logger.finish()
        if self.progress:
            self.progress_bar.fit_end()
