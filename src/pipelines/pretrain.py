"""MAE self-supervised pretraining pipeline for 2D CXR."""

import json
from pathlib import Path
from typing import Dict, List

import torch

from src.config.schema import AppConfig
from src.data.loaders import build_mae_loader, prepare_manifest_records
from src.model import MaskedAutoEncoder
from src.utils.metrics import aggregate_epoch_metrics
from src.utils.optim import build_optimizer, get_scheduler
from src.utils.seed import seed_everything


def _choose_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _run_epoch(loader, model, optimizer, device, train: bool) -> Dict[str, float]:
    model.train() if train else model.eval()
    batch_logs: List[Dict[str, float]] = []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            image = batch["image"].to(device).float()
            loss = model(image)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batch_logs.append({"loss": float(loss.item())})
    return aggregate_epoch_metrics(batch_logs)


def _save_encoder(model: MaskedAutoEncoder, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    torch.save(
        {"model_name": model.model_name, "encoder": model.export_encoder_state()},
        path,
    )
    return path


def run_pretrain(cfg: AppConfig) -> None:
    if cfg.mae is None:
        raise ValueError("mae config is required for pretraining")

    seed_everything(cfg.mae.seed)
    device = _choose_device(cfg.mae.use_gpu)
    out_dir = Path(cfg.mae.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = prepare_manifest_records(cfg.data)
    loader = build_mae_loader(records=records, data_cfg=cfg.data, mae_cfg=cfg.mae, split="train")

    model = MaskedAutoEncoder(
        model_name=cfg.model.model_name,
        pretrained=cfg.model.pretrained_imagenet,
        image_size=cfg.data.image_size,
        patch_size=cfg.mae.patch_size,
        mask_ratio=cfg.mae.mask_ratio,
    ).to(device)

    optimizer = build_optimizer(model, lr=cfg.mae.lr, wd=cfg.mae.weight_decay)
    scheduler = get_scheduler(
        optimizer=optimizer,
        warmup=cfg.mae.warmup_epochs,
        total=cfg.mae.epochs,
        unit="epoch",
    )

    history: List[Dict] = []
    best_loss = float("inf")
    best_ckpt = None

    for epoch in range(1, cfg.mae.epochs + 1):
        train_metrics = _run_epoch(loader, model, optimizer, device, train=True)
        scheduler.step()

        row = {"epoch": epoch, "train": train_metrics}
        history.append(row)
        print(f"[mae epoch {epoch}] loss={train_metrics.get('loss', 0):.4f}")

        if train_metrics.get("loss", float("inf")) < best_loss:
            best_loss = train_metrics["loss"]
            best_ckpt = _save_encoder(model, out_dir, "encoder_best.pt")

        if cfg.mae.save_every > 0 and (epoch % cfg.mae.save_every == 0):
            _save_encoder(model, out_dir, f"encoder_epoch_{epoch:03d}.pt")

    _save_encoder(model, out_dir, "encoder_last.pt")

    with (out_dir / "mae_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    if best_ckpt is not None:
        print(f"best MAE encoder: {best_ckpt}")
