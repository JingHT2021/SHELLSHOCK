"""Train a small PyTorch classifier for the fixed-font wind HUD digits."""
from __future__ import annotations

from shellshock.config.paths import DATA_ROOT

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from shellshock.perception.digit_model import WindDigitNet
from torch.utils.data import DataLoader, Dataset


class DigitDataset(Dataset):
    def __init__(self, rows, augment=False):
        self.rows = rows
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = cv2.imread(row["digit_path"], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(row["digit_path"])
        image = image.astype(np.float32) / 255.0
        if self.augment:
            if random.random() < 0.5:
                image = np.roll(image, random.choice([-1, 1]), axis=1)
            if random.random() < 0.4:
                image = np.clip(image * random.uniform(0.85, 1.15), 0, 1)
        return torch.from_numpy(image[None, ...]), int(row["digit_label"])


def read_manifest(path: Path, split: str):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == split]


def evaluate_predictions(rows, predictions):
    correct = sum(int(int(row["digit_label"]) == int(pred)) for row, pred in zip(rows, predictions))
    grouped = defaultdict(list)
    for row, pred in zip(rows, predictions):
        grouped[row["source_stem"]].append((int(row["position"]), int(pred), row["wind_value"]))
    exact = 0
    for items in grouped.values():
        items.sort()
        predicted = "".join(str(item[1]) for item in items)
        exact += int(int(predicted) == int(items[0][2]))
    return {"count": len(rows), "digit_accuracy": correct / len(rows) if rows else 0.0,
            "exact_value_accuracy": exact / len(grouped) if grouped else 0.0, "source_count": len(grouped)}


def predict(model, rows, device):
    model.eval()
    dataset = DigitDataset(rows)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    predictions = []
    confidences = []
    with torch.no_grad():
        for images, _ in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1)
            confidence, predicted = probabilities.max(dim=1)
            predictions.extend(predicted.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
    return predictions, confidences


def train(manifest: Path, output_dir: Path, seed: int = 42, epochs: int = 80) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = read_manifest(manifest, "train")
    validation_rows = read_manifest(manifest, "validation")
    test_rows = read_manifest(manifest, "test")
    model = WindDigitNet().to(device)
    loader = DataLoader(DigitDataset(train_rows, augment=True), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    counts = np.bincount([int(row["digit_label"]) for row in train_rows], minlength=10).astype(np.float32)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device))
    best_state = None
    best_score = -1.0
    patience = 12
    stale = 0
    for _ in range(epochs):
        model.train()
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(images.to(device)), labels.to(device)).backward()
            optimizer.step()
        val_predictions, _ = predict(model, validation_rows, device)
        score = evaluate_predictions(validation_rows, val_predictions)["exact_value_accuracy"]
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_predictions, test_confidences = predict(model, test_rows, device)
    metrics = evaluate_predictions(test_rows, test_predictions)
    metrics["low_confidence_count"] = sum(confidence < 0.75 for confidence in test_confidences)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "wind_digit_cnn.pt"
    torch.save(model.state_dict(), artifact)
    contract = {"model_type": "WindDigitNet", "input_size": [32, 48], "foreground": "dark",
                "threshold": 100, "num_classes": 10, "seed": seed, "confidence_threshold": 0.75,
                "artifact": str(artifact.resolve()), "device_at_train": str(device)}
    (output_dir / "contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    summary = {"device": str(device), "train_samples": len(train_rows), "validation_samples": len(validation_rows),
               "test_samples": len(test_rows), "metrics": metrics, "contract": contract}
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=(DATA_ROOT / 'annotate_check/wind_model_data/manifest.csv'))
    parser.add_argument("--output-dir", type=Path, default=(DATA_ROOT / 'runs/wind_digit_cnn'))
    parser.add_argument("--model", choices=["cnn", "both"], default="cnn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    print(json.dumps(train(args.manifest, args.output_dir, args.seed, args.epochs), indent=2))


if __name__ == "__main__":
    main()
