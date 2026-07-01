import math
from typing import Callable, Literal

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR


def _cosine_with_warmup_lambda(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def lr_lambda(step: int) -> float:
        step = max(0, int(step))
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def get_scheduler(
    optimizer: Optimizer,
    warmup: int,
    total: int,
    unit: Literal["epoch", "step"] = "step",
):
    warmup = max(0, int(warmup))
    total = max(1, int(total))
    if unit == "epoch":
        if warmup == 0:
            return CosineAnnealingLR(optimizer, T_max=total)
        if warmup >= total:
            return LinearLR(optimizer, start_factor=0.01, total_iters=total)
        scheduler_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total - warmup)
        return SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup])
    if unit == "step":
        return LambdaLR(optimizer, lr_lambda=_cosine_with_warmup_lambda(warmup, total))
    raise ValueError(f"Invalid unit: {unit}. Use 'epoch' or 'step'.")


def build_optimizer(net: torch.nn.Module, lr: float = 1e-4, wd: float = 1e-5) -> Optimizer:
    params = [p for p in net.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found in model.")
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
