import time

import torch
import torch.nn as nn

# from fastprogress.fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import format_time
from rich.progress import Progress

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
    # mb = master_bar(range(epochs))
    progress_bar = Progress()

    # mb.write(['epoch', 'train_loss', 'val loss', 'time', 'train time', 'val_time'], table=True)
    progress_bar.start()
    header = ['epoch', 'train_loss', 'val loss', 'time', 'train time', 'val_time']  # , 'val_time', 'train_time']
    progress_bar.print(' '.join([f"{value:>9}" for value in header]))
    main_job = progress_bar.add_task('fit...', total=epochs)
    # for epoch in mb:
    for epoch in range(epochs):
        progress_bar.tasks[main_job].description = f"ep {epoch + 1} of {epochs}"
        # mb.main_bar.comment = f"ep {epoch + 1} of {epochs}"

        model.train()
        start_time = time.time()
        # for batch_num, batch in enumerate(progress_bar(train_dl, parent=mb)):
        len_train_dl = len(train_dl)
        train_job = progress_bar.add_task('train', total=len_train_dl)
        for batch_num, batch in enumerate(train_dl):
            progress_bar._tasks[train_job].description = f"batch {batch_num}/{len_train_dl}"
            if debug_run and batch_num == debug_num_batches:
                break
            loss = loss_batch(model, loss_fn, batch)
            # mb.child.comment = f"loss {loss:0.4f}"
            loss.backward()
            opt.step()
            opt.zero_grad()
            progress_bar.update(train_job, advance=1)
        train_time = time.time() - start_time
        model.eval()
        len_val_dl = len(val_dl)
        val_job = progress_bar.add_task('validate...', total=len_val_dl)
        with torch.no_grad():
            valid_loss = []
            # for batch_num, batch in enumerate(progress_bar(val_dl, parent=mb)):
            for batch_num, batch in enumerate(val_dl):
                if debug_run and batch_num == debug_num_batches:
                    break
                valid_loss.append(loss_batch(model, loss_fn, batch).item())
            valid_loss = sum(valid_loss)
            progress_bar.update(val_job, advance=1)
        epoch_time = time.time() - start_time
        # mb.write([str(epoch + 1), f'{loss:0.4f}', f'{valid_loss / len(val_dl):0.4f}',
        #          format_time(epoch_time), format_time(train_time), format_time(epoch_time - train_time)], table=True)
        to_progress_bar = [f"{epoch + 1}", f"{loss:0.4f}", f"{valid_loss:0.4f}",
                           f"{format_time(epoch_time)}", f"{format_time(train_time)}"]  # , f"{format_time(val_time)}"]
        progress_bar.print(' '.join([f"{value:>9}" for value in to_progress_bar]))
        progress_bar.update(main_job, advance=1)
        progress_bar.remove_task(train_job)
        progress_bar.remove_task(val_job)

    progress_bar.remove_task(main_job)
    progress_bar.stop()


def loss_batch(model, loss_fn, batch):
    xb = batch[0].to(device)
    yb = batch[1].to(device)
    pred = model(xb)
    return loss_fn(pred, yb)
