"""Inference pipeline for 2D CXR BMD estimation."""

import csv
from pathlib import Path
from typing import Dict, List

import torch

from src.config.schema import AppConfig
from src.data.loaders import build_infer_loader, prepare_manifest_records
from src.model import FinetuneModel


def _choose_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_checkpoint(model: FinetuneModel, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)


def run_inference(cfg: AppConfig) -> None:
    if cfg.infer is None:
        raise ValueError("infer config is required")
    if not cfg.infer.checkpoint_path:
        raise ValueError("infer.checkpoint_path is required")

    device = _choose_device(cfg.infer.use_gpu)
    out_dir = Path(cfg.infer.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = prepare_manifest_records(cfg.data)
    infer_loader = build_infer_loader(
        records=records,
        data_cfg=cfg.data,
        infer_cfg=cfg.infer,
        regression=cfg.model.regression,
    )

    model = FinetuneModel(
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes,
        regression=cfg.model.regression,
        pretrained=False,
    ).to(device)
    _load_checkpoint(model, cfg.infer.checkpoint_path, device=device)
    model.eval()

    rows: List[Dict] = []
    with torch.no_grad():
        for batch in infer_loader:
            image = batch["image"].to(device).float()
            label = batch["label"].to(device).long()

            output = model(image)
            reg_out = None
            if model.regression:
                logits, reg_out = output
            else:
                logits = output

            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            for i in range(pred.shape[0]):
                row = {
                    "true_label": int(label[i].item()),
                    "pred_label": int(pred[i].item()),
                }
                for c in range(probs.shape[1]):
                    row[f"prob_{c}"] = float(probs[i, c].item())
                if reg_out is not None:
                    row["pred_t_score"] = float(reg_out[i].view(-1)[0].item())
                rows.append(row)

    output_csv = out_dir / f"predictions_{cfg.infer.split}.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            f.write("")
    print(f"saved inference result: {output_csv}")
