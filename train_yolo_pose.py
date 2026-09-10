"""Train and validate the ShellShock multi-class YOLO Pose model."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    model: str = "yolo11n-pose.pt"
    imgsz: int = 960
    epochs: int = 200
    batch: int = -1
    seed: int = 42
    workers: int = 4


def resolve_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "0" if torch.cuda.is_available() else "cpu"


def configure_ultralytics_config_dir(project_root: Path) -> Path:
    config_dir = Path(project_root) / "train" / "ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    return config_dir


def _metric_value(metrics: object, name: str) -> float | None:
    value = getattr(metrics, name, None)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _metric_summary(metrics: object | None) -> dict[str, float | None]:
    if metrics is None:
        return {name: None for name in ("precision", "recall", "mAP50", "mAP50_95")}
    return {
        "precision": _metric_value(metrics, "mp"),
        "recall": _metric_value(metrics, "mr"),
        "mAP50": _metric_value(metrics, "map50"),
        "mAP50_95": _metric_value(metrics, "map"),
    }


def train(data: Path, project: Path, name: str, config: TrainingConfig, device: str | None = None) -> dict[str, object]:
    configure_ultralytics_config_dir(Path.cwd())
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Ultralytics is not installed; install it with pip install ultralytics") from error
    selected_device = device or resolve_device()
    model = YOLO(config.model)
    model.train(
        data=str(Path(data).resolve()), imgsz=config.imgsz, epochs=config.epochs,
        batch=config.batch, seed=config.seed, device=selected_device,
        workers=config.workers, project=str(Path(project).resolve()), name=name,
        exist_ok=False, pretrained=True, plots=True, verbose=True,
    )
    run_dir = Path(project) / name
    best_path = run_dir / "weights" / "best.pt"
    validation = YOLO(str(best_path)).val(data=str(Path(data).resolve()), imgsz=config.imgsz, device=selected_device, plots=True)
    summary = {
        "config": asdict(config), "device": selected_device,
        "run_dir": str(run_dir.resolve()), "best_pt": str(best_path.resolve()),
        "last_pt": str((run_dir / "weights" / "last.pt").resolve()),
        "results_csv": str((run_dir / "results.csv").resolve()),
        "box_metrics": _metric_summary(getattr(validation, "box", None)),
        "pose_metrics": _metric_summary(getattr(validation, "pose", None)),
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("train/yolo_pose_dataset_v2/dataset.yaml"))
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--project", type=Path, default=Path("train/runs"))
    parser.add_argument("--name", default="shellshock_yolo11n_pose_v1")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    summary = train(args.data, args.project, args.name, TrainingConfig(args.model, args.imgsz, args.epochs, args.batch, args.seed, args.workers), args.device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
