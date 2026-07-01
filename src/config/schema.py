from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    manifest_path: str
    image_base_dir: str = ""
    img_path_key: str = "img_path"
    patient_id_key: str = "patient_id"
    label_key: str = "label"
    t_score_key: str = "t_score"
    split_key: str = "split"
    # Split behavior (used only when the manifest lacks a split column).
    generate_split: bool = True
    train_ratio: float = 0.7
    valid_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    # Image preprocessing.
    image_size: int = 224
    resize_size: int = 249  # 224 / 0.9, then center-cropped to image_size
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    # Augmentation (train split only).
    rand_flip_prob: float = 0.0
    rand_affine_prob: float = 0.5
    affine_degrees: float = 10.0
    affine_translate: float = 0.05


@dataclass
class ModelConfig:
    model_name: str = "resnet"
    num_classes: int = 3
    regression: bool = False
    pretrained_imagenet: bool = True


@dataclass
class MAEConfig:
    output_dir: str = "outputs/mae"
    epochs: int = 300
    batch_size: int = 16
    num_workers: int = 0
    lr: float = 1.5e-4
    weight_decay: float = 5e-2
    warmup_epochs: int = 10
    mask_ratio: float = 0.75
    patch_size: int = 16
    save_every: int = 50
    seed: int = 42
    use_gpu: bool = True


@dataclass
class TrainConfig:
    output_dir: str = "outputs/train"
    pretrained_ckpt: str = ""  # MAE encoder checkpoint; empty = ImageNet init only
    freeze_encoder: bool = False
    epochs: int = 100
    batch_size: int = 16
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 1
    cls_loss_weight: float = 1.0
    reg_loss_weight: float = 1.0
    save_every: int = 0
    seed: int = 42
    use_gpu: bool = True


@dataclass
class InferConfig:
    output_dir: str = "outputs/infer"
    checkpoint_path: str = ""
    batch_size: int = 16
    num_workers: int = 0
    split: str = "test"
    use_gpu: bool = True


@dataclass
class AppConfig:
    data: DataConfig
    model: ModelConfig
    mae: Optional[MAEConfig] = None
    train: Optional[TrainConfig] = None
    infer: Optional[InferConfig] = None

    def ensure_paths(self) -> None:
        Path(self.data.manifest_path)
        if self.mae is not None:
            Path(self.mae.output_dir).mkdir(parents=True, exist_ok=True)
        if self.train is not None:
            Path(self.train.output_dir).mkdir(parents=True, exist_ok=True)
        if self.infer is not None:
            Path(self.infer.output_dir).mkdir(parents=True, exist_ok=True)
