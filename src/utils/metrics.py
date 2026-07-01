from typing import Dict, List

import torch


def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def regression_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.abs(pred - target).mean().item())


def aggregate_epoch_metrics(history: List[Dict[str, float]]) -> Dict[str, float]:
    if not history:
        return {}
    keys = history[0].keys()
    out: Dict[str, float] = {}
    for key in keys:
        out[key] = sum(float(row.get(key, 0.0)) for row in history) / len(history)
    return out
