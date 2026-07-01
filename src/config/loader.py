"""Load YAML/JSON configs into typed dataclasses."""

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import yaml

from src.config.schema import (
    AppConfig,
    DataConfig,
    InferConfig,
    MAEConfig,
    ModelConfig,
    TrainConfig,
)

T = TypeVar("T")


def _read_raw(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    elif p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config extension: {p.suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}.")
    return data


def _filter_known(cls: Type[T], raw: Dict[str, Any]) -> Dict[str, Any]:
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return {k: v for k, v in raw.items() if k in known}


def _build(cls: Type[T], raw: Dict[str, Any]) -> T:
    return cls(**_filter_known(cls, raw))


def load_app_config(path: str, mode: str) -> AppConfig:
    raw = _read_raw(path)

    if "data" not in raw:
        raise ValueError("Config must contain a 'data' section.")
    if "model" not in raw:
        raise ValueError("Config must contain a 'model' section.")

    data = _build(DataConfig, raw["data"])
    model = _build(ModelConfig, raw["model"])

    mae = _build(MAEConfig, raw["mae"]) if "mae" in raw else None
    train = _build(TrainConfig, raw["train"]) if "train" in raw else None
    infer = _build(InferConfig, raw["infer"]) if "infer" in raw else None

    cfg = AppConfig(data=data, model=model, mae=mae, train=train, infer=infer)

    if mode == "pretrain" and cfg.mae is None:
        raise ValueError("mode='pretrain' requires a 'mae' section in the config.")
    if mode == "train" and cfg.train is None:
        raise ValueError("mode='train' requires a 'train' section in the config.")
    if mode == "infer" and cfg.infer is None:
        raise ValueError("mode='infer' requires an 'infer' section in the config.")

    cfg.ensure_paths()
    return cfg


def dump_effective_config(cfg: AppConfig, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
