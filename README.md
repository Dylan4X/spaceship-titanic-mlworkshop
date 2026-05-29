# AI3023 ML Workshop: Spaceship Titanic

Final project repository for Kaggle Spaceship Titanic. Our public-best XGBoost branch reached `0.81716` as team `EAP_Hater@MLW`.

## Main Files

- `notebooks/demo_081716_xgb.ipynb`: Kaggle demo notebook for the final submission.
- `notebooks/local_raw_data_demo.ipynb`: local notebook that starts from raw train/test data.
- `code/reproduce_081716_xgb.py`: script version of the same pipeline.
- `code/local_validation_reproduction.py`: consolidated local validation reproduction.
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

## Local Raw-Data Demo

For a local classroom demo based only on raw Kaggle files, open:

```text
notebooks/local_raw_data_demo.ipynb
```

It reads only `data/train.csv` and `data/test.csv`, then recomputes the dataset summary and EDA tables used in the slides.

## Local Experiments

To rerun the heavier validation reproduction script:

```powershell
python code\local_validation_reproduction.py
```

This retrains the local validation experiments and writes fresh summaries to `experiments/tables/local_validation_reproduction.csv`.
