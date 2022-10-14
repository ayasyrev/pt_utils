import time

import torch
import torch.nn as nn
from rich.progress import Progress

from .utils import format_time


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def validate(
    model: nn.Module,
    loss_fn,
    val_dl,
    device=device,
):
    model.to(device)
    progress_bar = Progress()

    progress_bar.start()
    start_time = time.time()
    model.eval()
    len_val_dl = len(val_dl)
    val_job = progress_bar.add_task("validate...", total=len_val_dl)
    with torch.no_grad():
        valid_losses = torch.tensor(0.0, device=device)

        for batch in val_dl:
            xb = batch[0].to(device)
            yb = batch[1].to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            valid_losses.add_(loss)
            progress_bar.update(val_job, advance=1)
        print(f"loss: {valid_losses.item() / len(val_dl.dataset):0.8f}")
        print(f"val time: {format_time(time.time() - start_time)}")
    progress_bar.stop()
