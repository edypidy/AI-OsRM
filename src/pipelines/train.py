"""Supervised BMD finetuning pipeline for 2D CXR."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.schema import AppConfig
from src.data.loaders import build_train_eval_loaders, prepare_manifest_records
from src.model import FinetuneModel
from src.utils.metrics import aggregate_epoch_metrics, classification_accuracy, regression_mae
from src.utils.optim import build_optimizer, get_scheduler
from src.utils.seed import seed_everything


def _choose_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_pretrained_encoder(model: FinetuneModel, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("encoder", ckpt)
    model.load_encoder_state(state, strict=True)
    print(f"loaded MAE encoder from {ckpt_path}")


def _run_epoch(
    loader: DataLoader,
    model: FinetuneModel,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    cls_criterion: nn.Module,
    reg_criterion: nn.Module,
    cls_loss_weight: float,
    reg_loss_weight: float,
    train: bool,
) -> Dict[str, float]:
    model.train() if train else model.eval()

    batch_logs: List[Dict[str, float]] = []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            image = batch["image"].to(device).float()
            label = batch["label"].to(device).long()
            t_score = batch.get("t_score")
            if t_score is not None:
                t_score = t_score.to(device).float().view(-1, 1)

            output = model(image)
            reg_out = None
            if model.regression:
                logits, reg_out = output
            else:
                logits = output

            cls_loss = cls_criterion(logits, label)
            total_loss = cls_loss_weight * cls_loss
            reg_loss = torch.tensor(0.0, device=device)
            if model.regression and reg_out is not None and t_score is not None:
                reg_loss = reg_criterion(reg_out, t_score)
                total_loss = total_loss + reg_loss_weight * reg_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

            log: Dict[str, float] = {
                "loss": float(total_loss.item()),
                "cls_loss": float(cls_loss.item()),
                "acc": classification_accuracy(logits.detach(), label.detach()),
                "reg_loss": float(reg_loss.item()),
            }
            if model.regression and reg_out is not None and t_score is not None:
                log["mae"] = regression_mae(reg_out.detach(), t_score.detach())
            batch_logs.append(log)
    return aggregate_epoch_metrics(batch_logs)


def _save_checkpoint(model: FinetuneModel, optimizer, epoch: int, out_dir: Path, name: str) -> Path:
    ckpt_path = out_dir / name
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_name": model.model_name,
        },
        ckpt_path,
    )
    return ckpt_path


def run_training(cfg: AppConfig) -> None:
    if cfg.train is None:
        raise ValueError("train config is required")

    seed_everything(cfg.train.seed)
    device = _choose_device(cfg.train.use_gpu)
    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = prepare_manifest_records(cfg.data)
    loaders = build_train_eval_loaders(
        records=records,
        data_cfg=cfg.data,
        train_cfg=cfg.train,
        regression=cfg.model.regression,
    )

    model = FinetuneModel(
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes,
        regression=cfg.model.regression,
        pretrained=cfg.model.pretrained_imagenet,
    ).to(device)

    if cfg.train.pretrained_ckpt:
        _load_pretrained_encoder(model, cfg.train.pretrained_ckpt, device)

    if cfg.train.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    optimizer = build_optimizer(model, lr=cfg.train.lr, wd=cfg.train.weight_decay)
    scheduler = get_scheduler(
        optimizer=optimizer,
        warmup=cfg.train.warmup_epochs,
        total=cfg.train.epochs,
        unit="epoch",
    )

    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()
    history: List[Dict] = []
    best_valid_loss = float("inf")
    best_ckpt: Optional[Path] = None

    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics = _run_epoch(
            loaders["train"],
            model=model,
            device=device,
            optimizer=optimizer,
            cls_criterion=cls_criterion,
            reg_criterion=reg_criterion,
            cls_loss_weight=cfg.train.cls_loss_weight,
            reg_loss_weight=cfg.train.reg_loss_weight,
            train=True,
        )
        valid_metrics = _run_epoch(
            loaders["valid"],
            model=model,
            device=device,
            optimizer=optimizer,
            cls_criterion=cls_criterion,
            reg_criterion=reg_criterion,
            cls_loss_weight=cfg.train.cls_loss_weight,
            reg_loss_weight=cfg.train.reg_loss_weight,
            train=False,
        )
        scheduler.step()

        history.append({"epoch": epoch, "train": train_metrics, "valid": valid_metrics})
        print(
            f"[epoch {epoch}] train_loss={train_metrics.get('loss', 0):.4f} "
            f"valid_loss={valid_metrics.get('loss', 0):.4f} "
            f"valid_acc={valid_metrics.get('acc', 0):.4f}"
        )

        if valid_metrics.get("loss", float("inf")) < best_valid_loss:
            best_valid_loss = valid_metrics["loss"]
            best_ckpt = _save_checkpoint(model, optimizer, epoch, out_dir, "best.pt")

        if cfg.train.save_every > 0 and (epoch % cfg.train.save_every == 0):
            _save_checkpoint(model, optimizer, epoch, out_dir, f"epoch_{epoch:03d}.pt")

    _save_checkpoint(model, optimizer, cfg.train.epochs, out_dir, "last.pt")

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    if best_ckpt is not None:
        print(f"best checkpoint: {best_ckpt}")
