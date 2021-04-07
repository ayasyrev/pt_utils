import time

import torch
import torch.nn as nn

import wandb

from fastprogress.fastprogress import master_bar, progress_bar, format_time


# class Logger:
#     def __init__(self) -> None:
#         pass


# class WandbLogger(Logger):
#     def __init__(self) -> None:
#         super().__init__()


class Learner:
    def __init__(self, model: nn.Module, loss_fn, opt_fn, train_dl, val_dl,
                 lr: float = 0.001, momentum: float = 0.9, prj_name: str = 'test',
                 logger: wandb = None, cfg: dict = None) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.opt_fn = opt_fn
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.lr = lr
        self.momentum = momentum
        self.opt = self.reset_opt()
        self.prj_name = prj_name

        if cfg is None:
            cfg = {'lr': self.lr}
        self.cfg = cfg
        if logger is None:
            self.logger = wandb.init(project=self.prj_name, config=self.cfg)
        self.logger.name = '-'.join([wandb.run.name.split('-')[-1], wandb.run.id])
        print(f"w_name: {self.logger.name}, num: {self.logger.name.split('-')[0]}")
        self.logger.watch(self.model, log='all')

    def reset_opt(self):
        return self.opt_fn(self.model.parameters(), lr=self.lr)

    def fit(self, epochs: int):
        train_start_time = time.time()
        self.model.to(self.device)
        mb = master_bar(range(epochs))
        mb.write(['epoch', 'train_loss', 'val loss', 'time', 'train time', 'val_time'], table=True)
        for epoch in mb:
            mb.main_bar.comment = f"ep {epoch + 1} of {epochs}"
            self.model.train()
            start_time = time.time()
            for batch_num, batch in enumerate(progress_bar(self.train_dl, parent=mb)):
                loss = self.loss_batch(batch)
                mb.child.comment = f"loss {loss:0.4f}"
                loss.backward()
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
                     format_time(epoch_time), format_time(train_time), format_time(epoch_time - train_time)],
                     table=True)
            self.logger.log({'epoch': epoch, 'train_loss': loss, 'val_loss': valid_loss})
            self.logger.log({'epoch': epoch, 'time': epoch_time,
                            'train_time': train_time, 'val_time': epoch_time - train_time})
        full_time = time.time() - train_start_time
        mb.write(f"full time: {format_time(full_time)}")
        self.logger.log({'full_time': full_time})
        self.logger.finish()

    def loss_batch(self, batch):
        xb = batch[0].to(self.device)
        yb = batch[1].to(self.device)
        pred = self.model(xb)
        return self.loss_fn(pred, yb)
