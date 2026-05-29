"""Compute and verify key numbers used in the PPT.

This script is intentionally quiet: it computes values from local data and
project evidence tables, checks them against the presentation anchors, and saves
machine-readable outputs for inspection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TABLE_DIR = PROJECT_ROOT / "experiments" / "tables"
OUTPUT_CSV = TABLE_DIR / "ppt_key_data_check.csv"
OUTPUT_JSON = TABLE_DIR / "ppt_key_data_check.json"


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(DATA_DIR / "train.csv"), pd.read_csv(DATA_DIR / "test.csv")


def compute_dataset_checks(train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    group_id = train["PassengerId"].str[:4]
    group_size = group_id.value_counts()
    group_labels = train.assign(GroupId=group_id).groupby("GroupId")["Transported"]

    return [
        {"section": "dataset", "metric": "training_rows", "computed": len(train), "ppt": 8693},
        {"section": "dataset", "metric": "test_rows", "computed": len(test), "ppt": 4277},
        {
            "section": "dataset",
            "metric": "raw_predictive_columns",
            "computed": train.drop(columns=["Transported"]).shape[1],
            "ppt": 13,
        },
        {"section": "dataset", "metric": "positive_rate", "computed": pct(train["Transported"].mean()), "ppt": "50.36%"},
        {
            "section": "dataset",
            "metric": "largest_missing_rate",
            "computed": pct(train.isna().mean().max()),
            "ppt": "2.50%",
        },
        {
            "section": "dataset",
            "metric": "multi_passenger_row_rate",
            "computed": pct(group_id.map(group_size).gt(1).mean(), 1),
            "ppt": "44.7%",
        },
        {
            "section": "dataset",
            "metric": "group_homogeneous_rate",
            "computed": pct(group_labels.nunique().eq(1).mean()),
            "ppt": "87.18%",
        },
    ]


def compute_eda_checks(train: pd.DataFrame) -> list[dict]:
    cabin = train["Cabin"].str.split("/", expand=True)
    frame = train.copy()
    frame["Deck"] = cabin[0]
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    frame["TotalSpend"] = frame[spend_cols].fillna(0).sum(axis=1)
    frame["NoSpend"] = frame["TotalSpend"].eq(0)

    deck_rates = frame.groupby("Deck")["Transported"].mean()
    planet_rates = frame.groupby("HomePlanet")["Transported"].mean()
    no_spend_rates = frame.groupby("Transported")["NoSpend"].mean()

    return [
        {"section": "eda", "metric": "deck_b_transport_rate", "computed": pct(deck_rates.loc["B"], 0), "ppt": "73%"},
        {"section": "eda", "metric": "deck_t_transport_rate", "computed": pct(deck_rates.loc["T"], 0), "ppt": "20%"},
        {
            "section": "eda",
            "metric": "europa_transport_rate",
            "computed": pct(planet_rates.loc["Europa"], 0),
            "ppt": "66%",
        },
        {
            "section": "eda",
            "metric": "earth_transport_rate",
            "computed": pct(planet_rates.loc["Earth"], 0),
            "ppt": "42%",
        },
        {
            "section": "eda",
            "metric": "no_spend_rate_if_transported",
            "computed": pct(no_spend_rates.loc[True], 1),
            "ppt": "spend signal",
        },
        {
            "section": "eda",
            "metric": "no_spend_rate_if_not_transported",
            "computed": pct(no_spend_rates.loc[False], 1),
            "ppt": "spend signal",
        },
    ]


def compute_table_checks() -> list[dict]:
    benchmark = pd.read_csv(TABLE_DIR / "model_benchmark.csv")
    ablation = pd.read_csv(TABLE_DIR / "feature_ablation.csv")
    tuning = pd.read_csv(TABLE_DIR / "tuning_summary.csv")
    cv_public = pd.read_csv(TABLE_DIR / "cv_vs_public.csv")
    xgb_recon = pd.read_csv(TABLE_DIR / "xgb_branch_reconstruction.csv")

    rows: list[dict] = []
    for model, ppt in [
        ("CatBoost", "0.8135"),
        ("XGBoost", "0.8106"),
        ("LightGBM", "0.8086"),
        ("Random Forest", "0.8062"),
        ("Logistic Regression", "0.7987"),
    ]:
        value = benchmark.loc[benchmark["model"].eq(model), "cv_mean_accuracy"].iloc[0]
        rows.append({"section": "benchmark", "metric": f"{model}_cv_accuracy", "computed": f"{value:.4f}", "ppt": ppt})

    for stage, ppt in [
        ("Raw official features", "0.7973"),
        ("+ Cabin features", "0.8125"),
        ("+ Group features", "0.8120"),
        ("+ Spend summaries", "0.8113"),
        ("+ Rule-consistency", "0.8097"),
    ]:
        value = ablation.loc[ablation["stage"].eq(stage), "mean_accuracy"].iloc[0]
        rows.append({"section": "ablation", "metric": stage, "computed": f"{value:.4f}", "ppt": ppt})

    for family, ppt in [("LightGBM", "0.0031"), ("CatBoost", "0.0023")]:
        value = tuning.loc[tuning["model_family"].eq(family), "audit_delta"].iloc[0]
        rows.append({"section": "tuning", "metric": f"{family}_audit_delta", "computed": f"{value:.4f}", "ppt": ppt})

    for label, local_ppt, public_ppt in [
        ("High-CV CatBoost candidate", "0.81905", "0.80593"),
        ("Public-best XGB branch", "0.80516", "0.81716"),
    ]:
        record = cv_public.loc[cv_public["label"].eq(label)].iloc[0]
        rows.append({"section": "cv_public", "metric": f"{label}_local", "computed": f"{record['local_cv']:.5f}", "ppt": local_ppt})
        rows.append({"section": "cv_public", "metric": f"{label}_public", "computed": f"{record['public_score']:.5f}", "ppt": public_ppt})

    for stage, ppt in [
        ("Compact preprocessing branch", "0.80445"),
        ("+ IsolationForest diagnostic", "0.80314"),
        ("+ Feature pruning", "0.80755"),
        ("Strict group-aware audit", "0.80516"),
    ]:
        value = xgb_recon.loc[xgb_recon["stage"].eq(stage), "accuracy"].iloc[0]
        rows.append({"section": "xgb_reconstruction", "metric": stage, "computed": f"{value:.5f}", "ppt": ppt})

    return rows


def mark_matches(rows: list[dict]) -> list[dict]:
    for row in rows:
        row["match"] = str(row["computed"]) == str(row["ppt"]) or row["ppt"] == "spend signal"
    return rows


def run_check() -> pd.DataFrame:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load_data()
    rows = compute_dataset_checks(train, test) + compute_eda_checks(train) + compute_table_checks()
    checked = pd.DataFrame(mark_matches(rows))
    checked.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_JSON.write_text(json.dumps(checked.to_dict(orient="records"), indent=2), encoding="utf-8")
    return checked


def main() -> None:
    checked = run_check()
    print(checked.to_string(index=False))


if __name__ == "__main__":
    main()
