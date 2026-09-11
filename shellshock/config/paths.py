"""Repository and data locations independent of the process working directory."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_data_root():
    if (PROJECT_ROOT / 'train').exists():
        return PROJECT_ROOT / 'train'
    # Linked worktrees reuse the original checkout's captures and trained models.
    git_file = PROJECT_ROOT / '.git'
    if git_file.is_file():
        git_dir = Path(git_file.read_text().strip().split(':', 1)[1].strip())
        if not git_dir.is_absolute():
            git_dir = PROJECT_ROOT / git_dir
        common_file = git_dir / 'commondir'
        if common_file.exists():
            common = (git_dir / common_file.read_text().strip()).resolve()
            return common.parent / 'train'
    return PROJECT_ROOT / 'train'


DATA_ROOT = Path(os.environ.get('SHELLSHOCK_DATA_ROOT', _default_data_root())).resolve()
YOLO_WEIGHTS = DATA_ROOT / 'runs/shellshock_yolo11n_pose_v1/weights/best.pt'
DIGIT_WEIGHTS = DATA_ROOT / 'runs/wind_digit_cnn_v2/wind_digit_cnn.pt'
LOG_ROOT = PROJECT_ROOT / 'logs'
