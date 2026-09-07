from train_yolo_fivefold import fold_metric_rows, make_run_names


def test_fold_run_names_are_unique_and_final_run_is_separate():
    names = make_run_names("shellshock_yolo11n_cv65")

    assert names["folds"] == [
        "shellshock_yolo11n_cv65_fold1",
        "shellshock_yolo11n_cv65_fold2",
        "shellshock_yolo11n_cv65_fold3",
        "shellshock_yolo11n_cv65_fold4",
        "shellshock_yolo11n_cv65_fold5",
    ]
    assert names["final"] == "shellshock_yolo11n_cv65_final_all"


def test_fold_metric_rows_only_contain_csv_columns():
    rows = fold_metric_rows(
        [{
            "fold": 1,
            "best_pt": "fold1/weights/best.pt",
            "metrics": {
                "precision": 0.9,
                "recall": 0.8,
                "mAP50": 0.7,
                "mAP50_95": 0.6,
                "per_class_ap50_95": {"0": 0.5},
            },
        }]
    )

    assert rows == [{
        "fold": 1,
        "precision": 0.9,
        "recall": 0.8,
        "mAP50": 0.7,
        "mAP50_95": 0.6,
        "best_pt": "fold1/weights/best.pt",
    }]
