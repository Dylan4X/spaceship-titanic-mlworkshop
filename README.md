# AI3023 ML Workshop: Spaceship Titanic

This repository contains the cleaned final code, report assets, experiment evidence, and reproducible demo for our AI3023 Machine Learning Workshop Spaceship Titanic project.

## Project Structure

- `code/`: runnable Python scripts for the final pipelines and supporting searches.
- `data/`: official Kaggle `train.csv`, `test.csv`, and `sample_submission.csv`.
- `notebooks/`: classroom demo notebook for the public-best compact XGBoost branch.
- `report/`: final IEEE-style paper, figures, tables, and scripts used to regenerate report assets.
- `experiments/`: preserved experiment notes and tables used by the paper.
- `submissions/`: selected representative submission CSVs, not every intermediate attempt.
- `references/`: original high-score reference notebook kept for auditability.

## Environment

Use Python 3.13 or a compatible recent Python 3 environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduce the XGBoost Demo Submission

This command runs compact preprocessing, a small Optuna search, the final fixed-parameter XGBoost training step, and writes a Kaggle submission.

```powershell
python code\reproduce_highscore_xgb.py --n-trials 8 --output submissions\Submission_XGB_demo_rerun.csv
```

Expected run characteristics on the prepared local environment:

- 27 processed features before pruning.
- 15 features after manual pruning.
- 8,666 rows retained by the IsolationForest diagnostic.
- 8,762 rows after KMeansSMOTE.
- Predicted true rate around `0.534`.

## Run the Demo Notebook

```powershell
python -m jupyter nbconvert --to notebook --execute notebooks\demo_repro_optuna_xgb_space_titanic.ipynb --output demo_repro_optuna_xgb_space_titanic.executed.ipynb --ExecutePreprocessor.timeout=900
```

The notebook is designed for teacher-facing presentation: it removes the original EDA-heavy clutter and keeps the training, tuning, and submission path visible.

## Report Evidence

The paper in `report/final_ieee_paper.pdf` separates two ideas:

- CatBoost is the clean scientific anchor under the validation-first benchmark.
- The compact XGBoost branch is the public-leaderboard best branch.

Key supporting tables are preserved in `experiments/tables/`:

- `model_benchmark.csv`
- `feature_ablation.csv`
- `tuning_summary.csv`
- `tuning_key_changes.csv`
- `xgb_parameter_provenance.csv`
- `xgb_branch_reconstruction.csv`
- `xgb_neighborhood_search.csv`
- `cv_vs_public.csv`
- `pipeline_audit.csv`

The XGBoost public-best branch should be explained as evidence-backed but not local-CV dominant: its parameters came from the original Optuna-labeled branch, the cleaned repo can rerun the pipeline, and the neighborhood search shows it sits in a reasonable tuned region.

## GitHub Submission Note

Before the final institutional submission, replace the GitHub placeholder inside `report/final_ieee_paper.tex` with the actual repository URL and rebuild/export the PDF if required by the instructor.
