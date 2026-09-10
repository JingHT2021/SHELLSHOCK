from train_yolo_pose import TrainingConfig, _metric_value, resolve_device


def test_pose_training_defaults_to_nano_pose_weights():
    assert TrainingConfig().model == "yolo11n-pose.pt"


def test_metric_value_returns_none_when_metric_is_unavailable():
    assert _metric_value(object(), "map") is None


def test_resolve_device_returns_cpu_when_cuda_is_unavailable(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorch())
    assert resolve_device() == "cpu"
