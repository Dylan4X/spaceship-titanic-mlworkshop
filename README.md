# AI3023 ML Workshop: Spaceship Titanic

Final project repository for Kaggle Spaceship Titanic. Our public-best XGBoost branch reached `0.81716` as team `EAP_Hater@MLW`.

## Main Files

- `notebooks/demo_081716_xgb.ipynb`: Kaggle demo notebook for the final submission.
- `code/reproduce_081716_xgb.py`: script version of the same pipeline.
- `report/final_ieee_paper.pdf`: final report.
- `experiments/tables/`: compact evidence tables used by the report.

## Kaggle Demo

Upload `notebooks/demo_081716_xgb.ipynb` to Kaggle, run all cells, and submit:

```text
/kaggle/working/Submission_XGB_0_81716.csv
```

## Local Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python code\reproduce_081716_xgb.py
```

The local script reads `data/train.csv`, `data/test.csv`, and `data/sample_submission.csv`, then writes:

```text
submissions/Submission_XGB_0_81716.csv
```

Expected prediction counts are approximately `{True: 2285, False: 1992}`.

## Note

Before final submission, replace `<team-github-repository-url>` in `report/final_ieee_paper.tex` and rebuild the PDF.
