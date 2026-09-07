"""Train the first ShellShock YOLO11n detector from a prepared dataset."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    model: str = "yolo11n.pt"
    imgsz: int = 960
    epochs: int = 200
    batch: int = -1
    seed: int = 42
    workers: int = 4
    lr0: float = 0.01


def resolve_device() -> str:
    """Return the first CUDA device when PyTorch exposes one, otherwise CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "0" if torch.cuda.is_available() else "cpu"


def configure_ultralytics_config_dir(project_root: Path) -> Path:
    """Keep Ultralytics' settings out of the user-profile Roaming directory."""
    config_dir = Path(project_root) / "train" / "ultralytics_config"
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    return config_dir


def _latest_results_row(results_csv: Path) -> dict[str, str]:
    if not results_csv.exists():
        return {}
    with results_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))[-1]


def train(data: Path, project: Path, name: str, config: TrainingConfig) -> dict[str, object]:
    configure_ultralytics_config_dir(Path.cwd())
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Ultralytics is not installed. Install it with `pip install ultralytics` before training.") from error
    device = resolve_device()
    model = YOLO(config.model)
    model.train(
        data=str(data.resolve()),
        imgsz=config.imgsz,
        epochs=config.epochs,
        batch=config.batch,
        seed=config.seed,
        device=device,
        workers=config.workers,
        lr0=config.lr0,
        project=str(project.resolve()),
        name=name,
        exist_ok=False,
        pretrained=True,
        plots=True,
        verbose=True,
    )
    run_dir = project / name
    best_path = run_dir / "weights" / "best.pt"
    validation = YOLO(str(best_path)).val(data=str(data.resolve()), imgsz=config.imgsz, device=device, plots=True)
    maps = getattr(validation.box, "maps", [])
    class_names = validation.names
    per_class_ap = {str(class_id): float(maps[class_id]) for class_id in range(len(maps)) if class_id in class_names}
    summary = {
        "config": asdict(config),
        "device": device,
        "run_dir": str(run_dir.resolve()),
        "best_pt": str(best_path.resolve()),
        "last_pt": str((run_dir / "weights" / "last.pt").resolve()),
        "results_csv": str((run_dir / "results.csv").resolve()),
        "metrics": {
            "precision": float(validation.box.mp),
            "recall": float(validation.box.mr),
            "mAP50": float(validation.box.map50),
            "mAP50_95": float(validation.box.map),
            "per_class_ap50_95": per_class_ap,
        },
        "last_results_row": _latest_results_row(run_dir / "results.csv"),
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("train/yolo_dataset/dataset.yaml"))
    parser.add_argument("--model", default="yolo11n.pt", help="Weights or model YAML to train from.")
    parser.add_argument("--project", type=Path, default=Path("train/runs"))
    parser.add_argument("--name", default="shellshock_yolo11n_v1")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train(args.data, args.project, args.name, TrainingConfig(model=args.model, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, seed=args.seed, workers=args.workers, lr0=args.lr0))
    metrics = summary["metrics"]
    print(f"Run directory: {summary['run_dir']}")
    print(f"best.pt: {summary['best_pt']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"mAP50: {metrics['mAP50']:.4f}")
    print(f"mAP50-95: {metrics['mAP50_95']:.4f}")


if __name__ == "__main__":
    main()
