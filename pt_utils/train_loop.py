import time

import torch
import torch.nn as nn

from fastprogress.fastprogress import master_bar, progress_bar, format_time


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def fit(
    epochs: int,
    lr: float,
    model: nn.Module,
    loss_fn,
    opt,
    train_dl,
    val_dl,
    debug_run: bool = False,
    debug_num_batches: int = 5
):
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    mb = master_bar(range(epochs))
    mb.write(['epoch', 'train_loss', 'val loss', 'time', 'train time', 'val_time'], table=True)
    for epoch in mb:
        mb.main_bar.comment = f"ep {epoch + 1} of {epochs}"
        model.train()
        start_time = time.time()
        for batch_num, batch in enumerate(progress_bar(train_dl, parent=mb)):
            if debug_run and batch_num == debug_num_batches:
                break
            loss = loss_batch(model, loss_fn, batch)
            mb.child.comment = f"loss {loss:0.4f}"
            loss.backward()
            opt.step()
            opt.zero_grad()
        train_time = time.time() - start_time
        model.eval()
        with torch.no_grad():
            valid_loss = []
            for batch_num, batch in enumerate(progress_bar(val_dl, parent=mb)):
                if debug_run and batch_num == debug_num_batches:
                    break
                valid_loss.append(loss_batch(model, loss_fn, batch).item())
            valid_loss = sum(valid_loss)
        epoch_time = time.time() - start_time
        mb.write([str(epoch + 1), f'{loss:0.4f}', f'{valid_loss / len(val_dl):0.4f}',
                 format_time(epoch_time), format_time(train_time), format_time(epoch_time - train_time)], table=True)


def loss_batch(model, loss_fn, batch):
    xb = batch[0].to(device)
    yb = batch[1].to(device)
    pred = model(xb)
    return loss_fn(pred, yb)
