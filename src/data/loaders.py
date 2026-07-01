"""Manifest reading, split handling, transforms and dataloader builders."""

import csv
import random
from typing import Dict, List, Optional, Tuple

from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from src.config.schema import DataConfig, InferConfig, MAEConfig, TrainConfig
from src.data.dataset import CXRDataset


def read_manifest(manifest_path: str) -> List[Dict]:
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = [dict(row) for row in reader]
    if not records:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return records


def _assign_split_from_ids(records: List[Dict], data_cfg: DataConfig) -> None:
    """Assign a split value to each record grouped by patient id (in place)."""
    id_key = data_cfg.patient_id_key
    use_ids = all(id_key in r and r[id_key] not in (None, "") for r in records)

    if use_ids:
        groups: Dict[str, List[Dict]] = {}
        for r in records:
            groups.setdefault(str(r[id_key]), []).append(r)
        keys = sorted(groups.keys())
    else:
        groups = {str(i): [r] for i, r in enumerate(records)}
        keys = sorted(groups.keys(), key=int)

    rng = random.Random(data_cfg.seed)
    rng.shuffle(keys)

    n = len(keys)
    n_train = int(n * data_cfg.train_ratio)
    n_valid = int(n * data_cfg.valid_ratio)
    train_keys = set(keys[:n_train])
    valid_keys = set(keys[n_train : n_train + n_valid])

    for key in keys:
        if key in train_keys:
            split = "train"
        elif key in valid_keys:
            split = "valid"
        else:
            split = "test"
        for r in groups[key]:
            r[data_cfg.split_key] = split


def prepare_manifest_records(data_cfg: DataConfig) -> List[Dict]:
    records = read_manifest(data_cfg.manifest_path)
    has_split = all(
        data_cfg.split_key in r and r[data_cfg.split_key] not in (None, "") for r in records
    )
    if not has_split:
        if not data_cfg.generate_split:
            raise ValueError(
                f"Manifest lacks split column '{data_cfg.split_key}' and "
                "data.generate_split is false."
            )
        _assign_split_from_ids(records, data_cfg)
    return records


def filter_by_split(records: List[Dict], split: str, split_key: str) -> List[Dict]:
    return [r for r in records if str(r.get(split_key)) == split]


def build_transforms(data_cfg: DataConfig, train: bool) -> T.Compose:
    normalize = T.Normalize(mean=data_cfg.normalize_mean, std=data_cfg.normalize_std)
    if train:
        ops = []
        if data_cfg.rand_affine_prob > 0:
            ops.append(
                T.RandomApply(
                    [
                        T.RandomAffine(
                            degrees=data_cfg.affine_degrees,
                            translate=(data_cfg.affine_translate, data_cfg.affine_translate),
                        )
                    ],
                    p=data_cfg.rand_affine_prob,
                )
            )
        if data_cfg.rand_flip_prob > 0:
            ops.append(T.RandomHorizontalFlip(p=data_cfg.rand_flip_prob))
        ops.append(
            T.RandomResizedCrop(
                size=(data_cfg.image_size, data_cfg.image_size),
                scale=(0.9, 1.0),
                ratio=(1.0, 1.0),
                antialias=True,
            )
        )
        ops.append(normalize)
        return T.Compose(ops)

    return T.Compose(
        [
            T.Resize(size=(data_cfg.resize_size, data_cfg.resize_size), antialias=True),
            T.CenterCrop(size=(data_cfg.image_size, data_cfg.image_size)),
            normalize,
        ]
    )


def build_dataloader(
    records: List[Dict],
    data_cfg: DataConfig,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    train: bool,
    task: str,
    regression: bool,
) -> Tuple[DataLoader, CXRDataset]:
    transforms = build_transforms(data_cfg, train=train)
    dataset = CXRDataset(
        records=records,
        base_path=data_cfg.image_base_dir,
        transforms=transforms,
        task=task,
        img_path_key=data_cfg.img_path_key,
        label_key=data_cfg.label_key,
        t_score_key=data_cfg.t_score_key,
        regression=regression,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=train,
    )
    return loader, dataset


def build_train_eval_loaders(
    records: List[Dict], data_cfg: DataConfig, train_cfg: TrainConfig, regression: bool
) -> Dict[str, DataLoader]:
    train_records = filter_by_split(records, "train", data_cfg.split_key)
    valid_records = filter_by_split(records, "valid", data_cfg.split_key)
    if not train_records:
        raise ValueError("No records found for split 'train'.")
    if not valid_records:
        raise ValueError("No records found for split 'valid'.")

    train_loader, _ = build_dataloader(
        train_records,
        data_cfg=data_cfg,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        shuffle=True,
        train=True,
        task="supervised",
        regression=regression,
    )
    valid_loader, _ = build_dataloader(
        valid_records,
        data_cfg=data_cfg,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        shuffle=False,
        train=False,
        task="supervised",
        regression=regression,
    )
    return {"train": train_loader, "valid": valid_loader}


def build_mae_loader(
    records: List[Dict], data_cfg: DataConfig, mae_cfg: MAEConfig, split: Optional[str] = "train"
) -> DataLoader:
    if split is not None:
        subset = filter_by_split(records, split, data_cfg.split_key)
        if not subset:
            subset = records
    else:
        subset = records
    loader, _ = build_dataloader(
        subset,
        data_cfg=data_cfg,
        batch_size=mae_cfg.batch_size,
        num_workers=mae_cfg.num_workers,
        shuffle=True,
        train=True,
        task="mae",
        regression=False,
    )
    return loader


def build_infer_loader(
    records: List[Dict], data_cfg: DataConfig, infer_cfg: InferConfig, regression: bool
) -> DataLoader:
    split_records = filter_by_split(records, infer_cfg.split, data_cfg.split_key)
    if not split_records:
        raise ValueError(f"No records found for split '{infer_cfg.split}'.")
    loader, _ = build_dataloader(
        split_records,
        data_cfg=data_cfg,
        batch_size=infer_cfg.batch_size,
        num_workers=infer_cfg.num_workers,
        shuffle=False,
        train=False,
        task="supervised",
        regression=regression,
    )
    return loader
