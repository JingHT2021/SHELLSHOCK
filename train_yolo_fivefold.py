"""Run five-fold ShellShock YOLO fine-tunes and a separate all-data final model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

from train_yolo import TrainingConfig, train


def make_run_names(prefix: str) -> dict[str, object]:
    return {"folds": [f"{prefix}_fold{index}" for index in range(1, 6)], "final": f"{prefix}_final_all"}


def fold_metric_rows(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return only the fixed aggregate columns, excluding nested per-class metrics."""
    metric_names = ("precision", "recall", "mAP50", "mAP50_95")
    return [
        {
            "fold": summary["fold"],
            **{metric: summary["metrics"][metric] for metric in metric_names},
            "best_pt": summary["best_pt"],
        }
        for summary in summaries
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=Path, default=Path("train/yolo_fivefold"))
    parser.add_argument("--base-model", default="train/runs/shellshock_yolo11n_v1/weights/best.pt")
    parser.add_argument("--project", type=Path, default=Path("train/runs"))
    parser.add_argument("--report-dir", type=Path, default=Path("train/fivefold"))
    parser.add_argument("--prefix", default="shellshock_yolo11n_cv65")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--skip-folds",
        action="store_true",
        help="reuse completed fold training_summary.json files and only write reports/final model",
    )
    args = parser.parse_args()
    names = make_run_names(args.prefix)
    config = TrainingConfig(model=args.base_model, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, lr0=args.lr0, seed=args.seed, workers=args.workers)
    args.report_dir.mkdir(parents=True, exist_ok=False)
    summaries = []
    for index, run_name in enumerate(names["folds"], start=1):
        if args.skip_folds:
            summary_path = args.project / run_name / "training_summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"cannot skip unfinished fold {index}: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = train(args.datasets_dir / f"fold{index}" / "dataset.yaml", args.project, run_name, config)
        summary["fold"] = index
        summaries.append(summary)
    metrics = ["precision", "recall", "mAP50", "mAP50_95"]
    aggregate = {metric: {"mean": mean(item["metrics"][metric] for item in summaries), "std": pstdev(item["metrics"][metric] for item in summaries)} for metric in metrics}
    with (args.report_dir / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("fold", *metrics, "best_pt"))
        writer.writeheader()
        writer.writerows(fold_metric_rows(summaries))
    report = {"config": config.__dict__, "folds": summaries, "cross_validation": aggregate}
    (args.report_dir / "cross_validation_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    final_summary = train(args.datasets_dir / "final_all" / "dataset.yaml", args.project, names["final"], config)
    (args.report_dir / "final_all_summary.json").write_text(json.dumps(final_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cross_validation": aggregate, "final_best_pt": final_summary["best_pt"]}, indent=2))


if __name__ == "__main__":
    main()
