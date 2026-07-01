import argparse
from pathlib import Path

from src.config.loader import dump_effective_config, load_app_config
from src.pipelines.infer import run_inference
from src.pipelines.pretrain import run_pretrain
from src.pipelines.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D CXR BMD training/inference entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain_parser = subparsers.add_parser("pretrain", help="Run MAE self-supervised pretraining")
    pretrain_parser.add_argument("--config", required=True, help="Path to MAE config (.yaml/.json)")

    train_parser = subparsers.add_parser("train", help="Run supervised BMD finetuning")
    train_parser.add_argument("--config", required=True, help="Path to train config (.yaml/.json)")

    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--config", required=True, help="Path to infer config (.yaml/.json)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "pretrain":
        cfg = load_app_config(args.config, mode="pretrain")
        dump_effective_config(cfg, str(Path(cfg.mae.output_dir) / "effective_config.json"))
        run_pretrain(cfg)
        return

    if args.command == "train":
        cfg = load_app_config(args.config, mode="train")
        dump_effective_config(cfg, str(Path(cfg.train.output_dir) / "effective_config.json"))
        run_training(cfg)
        return

    if args.command == "infer":
        cfg = load_app_config(args.config, mode="infer")
        dump_effective_config(cfg, str(Path(cfg.infer.output_dir) / "effective_config.json"))
        run_inference(cfg)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
