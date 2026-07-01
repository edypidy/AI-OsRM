# AI-BMD

Repository for **"Artificial Intelligence–Based Opportunistic Screening for Osteoporosis on Chest Radiographs: Development and Multicenter Validation"**

AI-BMD estimates bone mineral density (BMD) status from routine 2D chest radiographs (CXR). Training follows a two-stage recipe: self-supervised **masked autoencoder (MAE) pretraining**, followed by **supervised finetuning** for osteoporosis estimation with an optional continuous BMD/T-score regression head.

## Overview

```
CXR image ──▶ MAE pretraining (self-supervised)
                     │  encoder weights
                     ▼
             Supervised finetuning ──▶ classification
                                       (+ optional T-score regression)
                     │  trained model
                     ▼
                 Inference ──▶ predictions.csv
```

- Backbones: `resnet`, `densenet` (optional: `convnext`, `vit`, `swin`).
- The MAE encoder is reused for finetuning; a lightweight convolutional decoder is used only during pretraining.

## Installation

```bash
git clone <repository-url>
cd BMD_github
pip install -r requirements.txt
```

Requires Python 3.9+ and a CUDA-capable GPU (recommended).

## Data

Provide a CSV manifest. Only `img_path` is required for MAE pretraining; `label` (and optionally `t_score`) are required for finetuning.

| column       | description                                        |
| ------------ | -------------------------------------------------- |
| `img_path`   | path to the CXR image (relative to `image_base_dir`) |
| `label`      | integer class id (`0..num_classes-1`)              |
| `t_score`    | continuous BMD / T-score (only if `regression: true`) |
| `patient_id` | patient id used for patient-level split (optional) |
| `split`      | `train` / `valid` / `test` (auto-generated if absent) |

If `split` is missing, splits are generated at the patient level using `train_ratio` / `valid_ratio` / `test_ratio`.

## Usage

```bash
# 1) MAE self-supervised pretraining
python main.py pretrain --config configs/mae.example.yaml

# 2) Supervised BMD finetuning (loads the MAE encoder)
python main.py train --config configs/train.example.yaml

# 3) Inference
python main.py infer --config configs/infer.example.yaml
```

Key settings live in the YAML configs under `configs/`:

- `model.model_name`: backbone (`resnet` / `densenet` / `convnext` / `vit` / `swin`).
- `model.num_classes`: `3` for 3-class, `2` for binary.
- `model.regression`: enable the continuous T-score head.
- `train.pretrained_ckpt`: path to the MAE encoder checkpoint (e.g. `outputs/mae/encoder_best.pt`).

Outputs (checkpoints, `history.json`, and `predictions_{split}.csv`) are written to each stage's `output_dir`.

## Repository structure

```
BMD_github/
├── main.py                     # pretrain | train | infer entrypoint
├── configs/                    # example YAML configs
├── src/
│   ├── config/                 # config schema & loader
│   ├── data/                   # CSV manifest, dataset, dataloaders
│   ├── model/                  # MAE + finetune model factory
│   ├── pipelines/              # pretrain / train / infer loops
│   └── utils/                  # seeding, optimizer/scheduler, metrics
└── requirements.txt
```

## Citation
On the revision process.
