# AI3023 ML Workshop: Spaceship Titanic

This repository contains the final report, supporting experiment evidence, and the reproducible XGBoost pipeline for our AI3023 Machine Learning Workshop project on Kaggle Spaceship Titanic.

The public-best branch reached a Kaggle public score of `0.81716` for team `EAP_Hater@MLW`.

## Repository Layout

- `code/reproduce_081716_xgb.py`: script version of the final public-best XGBoost pipeline.
- `notebooks/demo_081716_xgb.ipynb`: Kaggle-first demo notebook for reproducing the final submission.
- `data/`: official Kaggle CSV files for local reproduction.
- `report/`: final IEEE-style report, PDF, and figures.
- `experiments/tables/`: compact experiment evidence used by the report.

Older search scripts, scratch notebooks, and intermediate submissions are intentionally removed so the repository stays focused and easy to review.

## Environment

Use Python 3.13 or a compatible recent Python 3 environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduce the Final Submission on Kaggle

Recommended for demo and grading:

1. Open Kaggle for the Spaceship Titanic competition.
2. Upload or paste `notebooks/demo_081716_xgb.ipynb`.
3. Run all cells.
4. Submit `/kaggle/working/Submission_XGB_0_81716.csv`.

The final XGBoost configuration is:

- Seed: `36086`
- Resampling: `KMeansSMOTE(sampling_strategy=1, n_jobs=-1)`
- `lambda=3.06`
- `alpha=4.582`
- `colsample_bytree=0.93`
- `subsample=0.96`
- `n_estimators=950`
- `max_depth=5`
- `learning_rate=0.0475`

## Reproduce Locally

The script automatically finds the CSV files in either `data/` or the current directory.

```powershell
python code\reproduce_081716_xgb.py
```

Expected local output:

- `submissions/Submission_XGB_0_81716.csv`
- Resampled class counts close to `{0: 4384, 1: 4378}`
- Prediction counts close to `{True: 2285, False: 1992}`

## Report Evidence

The report separates two claims:

- The clean CatBoost line is the strongest validation-first scientific anchor.
- The compact XGBoost branch is the public-best competition branch.

Supporting evidence is preserved in `experiments/tables/`, including model benchmarks, feature ablation, tuning summaries, CV-vs-public comparisons, and XGBoost branch reconstruction.

## Final Report

- PDF: `report/final_ieee_paper.pdf`
- Source: `report/final_ieee_paper.tex`

Before final submission, replace `<team-github-repository-url>` in `report/final_ieee_paper.tex` with the actual GitHub URL and rebuild the PDF.
