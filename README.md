# AI3023 ML Workshop: Spaceship Titanic

This repository contains the cleaned final code, report assets, experiment evidence, and reproducible demo for our AI3023 Machine Learning Workshop Spaceship Titanic project.

## Project Structure

- `code/`: runnable Python scripts for the final pipelines and supporting searches.
- `data/`: official Kaggle `train.csv`, `test.csv`, and `sample_submission.csv`.
- `notebooks/`: classroom demo notebook for the original `0-814` Optuna XGBoost branch.
- `report/`: final IEEE-style paper, figures, tables, and scripts used to regenerate report assets.
- `experiments/`: preserved experiment notes and tables used by the paper.
- `submissions/`: selected representative submission CSVs, not every intermediate attempt.

## Environment

Use Python 3.13 or a compatible recent Python 3 environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduce the 0.814 XGBoost Branch

This is the strict reproduction path for `0-814-optuna-xgb-space-titanic.ipynb`. It keeps the original feature engineering, fixed Optuna-labeled XGBoost parameters, manual feature drop list, KMeansSMOTE step, and final submission export.

The script automatically finds the official CSVs if they are either in `data/` or directly in the project root.

```powershell
python code\reproduce_0_814_notebook_exact.py
```

Expected run characteristics on the prepared local environment:

- 27 processed features before pruning.
- 15 features after manual pruning.
- KMeansSMOTE class counts close to the original executed notebook's `4384` and `4378`; because the original SMOTE call is unseeded, reruns can differ slightly.
- A generated submission at `submissions/Submission_XGB_exact.csv`.
- A metrics/provenance file at `experiments/tables/notebook_0814_exact_run_metrics.json`.

Important reproducibility note: the original notebook leaves `shuffle`, `KMeansSMOTE`, and the final `XGBClassifier` randomness unseeded. Therefore, different machines or package builds can produce small row-level differences. The repository includes `submissions/submission_xgb_reference_0814.csv` as the local highest-score reference CSV, and the exact script/notebook reports row-level differences against it instead of hiding the variance.

## Run the Demo Notebook

```powershell
python -m jupyter nbconvert --to notebook --execute notebooks\0_814_exact_reproduction_demo.ipynb --output 0_814_exact_reproduction_demo.executed.ipynb --ExecutePreprocessor.timeout=900
```

For a VSCode classroom demo, open `notebooks/0_814_exact_reproduction_demo.ipynb` and run from the first cell. The first cell installs `requirements.txt` into the active kernel, which fixes the common `ModuleNotFoundError: No module named 'imblearn'` issue when VSCode is using a different Python environment.

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
