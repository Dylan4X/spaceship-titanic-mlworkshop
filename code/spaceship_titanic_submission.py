from pathlib import Path
import shutil

from project_paths import EXPERIMENTS_TABLE_DIR, SUBMISSIONS_DIR
from spaceship_titanic_pycaret_catboost import main as build_best_submission


ROOT = Path(__file__).resolve().parent
BEST_SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_catboost_pycaret_exact.csv"
DEFAULT_SUBMISSION_PATH = SUBMISSIONS_DIR / "submission.csv"
BEST_RESULTS_PATH = EXPERIMENTS_TABLE_DIR / "pycaret_catboost_results.json"
DEFAULT_RESULTS_PATH = EXPERIMENTS_TABLE_DIR / "cv_metrics.json"


def main() -> None:
    # Keep the generic entrypoint pinned to the best clean public-LB pipeline.
    build_best_submission()
    shutil.copyfile(BEST_SUBMISSION_PATH, DEFAULT_SUBMISSION_PATH)
    shutil.copyfile(BEST_RESULTS_PATH, DEFAULT_RESULTS_PATH)


if __name__ == "__main__":
    main()
