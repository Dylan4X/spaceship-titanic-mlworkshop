"""Shared repository paths for runnable scripts."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_data_dir(start: Path = PROJECT_ROOT) -> Path:
    """Find official Kaggle CSVs in a portable project layout."""
    candidates = [
        start / "data",
        start,
        Path.cwd() / "data",
        Path.cwd(),
    ]
    for path in candidates:
        if (path / "train.csv").exists() and (path / "test.csv").exists():
            return path
    return start / "data"


DATA_DIR = find_data_dir()
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_TABLE_DIR = EXPERIMENTS_DIR / "tables"

SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
