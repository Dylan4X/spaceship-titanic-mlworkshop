"""Shared repository paths for runnable scripts."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_TABLE_DIR = EXPERIMENTS_DIR / "tables"

SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
