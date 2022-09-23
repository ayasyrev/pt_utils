from typing import Sequence

import torch


def accuracy(
    outputs: torch.Tensor, targets: torch.Tensor, topk: Sequence[int] = (1,)
) -> Sequence[torch.Tensor]:
    """
    Computes multiclass accuracy@topk for the specified values of `topk`.
    """
    max_k = max(topk)
    batch_size = targets.size(0)

    if len(outputs.shape) == 1 or outputs.shape[1] == 1:
        # binary accuracy
        pred = outputs.t()
    else:
        # multiclass accuracy
        _, pred = outputs.topk(max_k, 1, True, True)
        pred = pred.t()
    correct = pred.eq(targets.long().view(1, -1).expand_as(pred))

    output = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        output.append(correct_k.mul_(1.0 / batch_size))
    return output
