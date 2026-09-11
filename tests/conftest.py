"""Use ordinary inherited Windows ACLs for test artifact directories."""
from pathlib import Path
from uuid import uuid4
import pytest

@pytest.fixture
def tmp_path():
    path=Path(__file__).resolve().parents[1]/'.codex_tmp'/'cases'/uuid4().hex
    path.mkdir(parents=True)
    return path
