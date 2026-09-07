from train_yolo import TrainingConfig, configure_ultralytics_config_dir


def test_default_training_config_preserves_requested_reproducible_defaults():
    config = TrainingConfig()
    assert config.model == "yolo11n.pt"
    assert config.imgsz == 960
    assert config.epochs == 200
    assert config.seed == 42


def test_training_config_allows_a_low_learning_rate_finetune_from_best_weights():
    config = TrainingConfig(
        model="train/runs/shellshock_yolo11n_v1/weights/best.pt",
        epochs=40,
        lr0=0.001,
    )

    assert config.model.endswith("weights/best.pt")
    assert config.epochs == 40
    assert config.lr0 == 0.001


def test_ultralytics_config_is_kept_inside_project_train_directory(monkeypatch, tmp_path):
    configure_ultralytics_config_dir(tmp_path)
    assert __import__("os").environ["YOLO_CONFIG_DIR"] == str(tmp_path / "train" / "ultralytics_config")
