from dataclasses import dataclass, field
import time

import torch
import torch.nn as nn

from fastprogress.fastprogress import master_bar, progress_bar, format_time

from accelerate import Accelerator

from pt_utils.logger import LoggerCfg, Logger
from local_utils.pt_utils.transforms import Norm


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
                 logger: Logger = None) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.opt_fn = opt_fn
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.cfg = cfg

        self.device = device
        self.opt = self.reset_opt()

        self.batch_tfm = Norm(device=self.device)

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
        mb = master_bar(range(epochs))
        mb.write(['epoch', 'train_loss', 'val loss', 'time', 'train time', 'val_time', 'val_time', 'train_time'],
                 table=True)
        for epoch in mb:
            mb.main_bar.comment = f"ep {epoch + 1} of {epochs}"
            self.model.train()
            start_time = time.time()
            for batch_num, batch in enumerate(progress_bar(self.train_dl, parent=mb)):
                loss = self.loss_batch(batch)
                mb.child.comment = f"loss {loss:0.4f}"
                accelerator.backward(loss)
                self.opt.step()
                self.opt.zero_grad()
            train_time = time.time() - start_time
            self.model.eval()
            with torch.no_grad():
                valid_losses = []
                for batch_num, batch in enumerate(progress_bar(self.val_dl, parent=mb)):
                    valid_losses.append(self.loss_batch(batch).item())
                valid_loss = sum(valid_losses) / len(valid_losses)
            epoch_time = time.time() - start_time
            mb.write([str(epoch + 1), f'{loss:0.4f}', f'{valid_loss:0.4f}',
                     format_time(epoch_time), format_time(train_time), format_time(epoch_time - train_time),
                     f"{(epoch_time - train_time):0.4f}", f"{train_time:0.4f}"], table=True)
            self.logger.log({'epoch': epoch, 'train_loss': loss, 'val_loss': valid_loss,
                            'time': epoch_time, 'train_time': train_time, 'val_time': epoch_time - train_time})
        full_time = time.time() - train_start_time
        mb.write(f"full time: {format_time(full_time)}")
        self.logger.log({'full_time': full_time})
        self.logger.finish()

    def loss_batch(self, batch):
        input = self.batch_tfm(batch[0])
        pred = self.model(input)
        return self.loss_fn(pred, batch[1])
