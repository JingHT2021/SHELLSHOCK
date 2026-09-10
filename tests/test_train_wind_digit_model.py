import csv
import json
from pathlib import Path
import shutil

import train_wind_digit_model as module


def test_evaluate_reports_digit_and_string_accuracy():
    rows = [
        {"source_stem": "scene", "digit_label": "6", "position": "0", "wind_value": "64"},
        {"source_stem": "scene", "digit_label": "4", "position": "1", "wind_value": "64"},
    ]
    metrics = module.evaluate_predictions(rows, [6, 4])
    assert metrics["count"] == 2
    assert metrics["digit_accuracy"] == 1.0
    assert metrics["exact_value_accuracy"] == 1.0


def test_model_artifact_contains_preprocessing_contract():
    output = Path("tests/.wind_model_contract_test")
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    contract = {"input_size": [32, 48], "foreground": "dark", "num_classes": 10, "seed": 42}
    (output / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
    loaded = json.loads((output / "contract.json").read_text(encoding="utf-8"))
    assert loaded["input_size"] == [32, 48]
    assert loaded["foreground"] == "dark"
    assert loaded["num_classes"] == 10
    assert loaded["seed"] == 42
