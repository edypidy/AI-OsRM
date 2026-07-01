"""Dataset for 2D chest X-ray BMD estimation."""

from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image


class CXRDataset(Dataset):
    """2D CXR dataset.

    Each record is a dict parsed from the CSV manifest. Two tasks are supported:

    * ``task="mae"``: returns ``{"image": Tensor}`` only (self-supervised).
    * ``task="supervised"``: returns ``{"image", "label", ("t_score")}``.

    Images are read as 3-channel tensors, scaled to [0, 1]; further
    normalization/augmentation is delegated to ``transforms``.
    """

    def __init__(
        self,
        records: List[Dict],
        base_path: str = "",
        transforms: Optional[Callable] = None,
        task: str = "supervised",
        img_path_key: str = "img_path",
        label_key: str = "label",
        t_score_key: str = "t_score",
        regression: bool = False,
        return_path: bool = False,
    ) -> None:
        if task not in ("mae", "supervised"):
            raise ValueError(f"Unsupported task '{task}'. Use 'mae' or 'supervised'.")
        self.records = records
        self.base_path = base_path
        self.transforms = transforms
        self.task = task
        self.img_path_key = img_path_key
        self.label_key = label_key
        self.t_score_key = t_score_key
        self.regression = regression
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        full_path = str(Path(self.base_path) / rel_path) if self.base_path else rel_path
        image = read_image(full_path, mode=ImageReadMode.RGB).float()
        image -= image.min()
        image /= image.max().clamp_min(1e-6)
        if self.transforms is not None:
            image = self.transforms(image)
        return image

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        image = self._load_image(str(record[self.img_path_key]))

        sample: Dict = {"image": image}
        if self.task == "supervised":
            sample["label"] = torch.tensor(int(record[self.label_key]), dtype=torch.long)
            if self.regression:
                t_score = record.get(self.t_score_key)
                if t_score is None or t_score == "":
                    raise ValueError(
                        f"regression=True but '{self.t_score_key}' is missing for record {idx}."
                    )
                sample["t_score"] = torch.tensor(float(t_score), dtype=torch.float32)

        if self.return_path:
            sample["path"] = str(record[self.img_path_key])
        return sample
