"""Line-by-line reproduction of the original 0.814 Optuna XGB notebook branch.

This script intentionally mirrors `0-814-optuna-xgb-space-titanic.ipynb` rather
than the cleaned research pipeline. The original notebook did not fix every
random source, so the generated CSV can vary slightly across runs and package
versions. When a reference highest-score CSV is present, this script reports the
row-level difference instead of hiding that variance.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import KMeansSMOTE
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle


warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PARAMS_XGB_BEST = {
    "lambda": 3.06,
    "alpha": 4.582,
    "colsample_bytree": 0.93,
    "subsample": 0.96,
    "n_estimators": 725,
    "max_depth": 5,
    "learning_rate": 0.05,
}

DROP_LIST = [
    "ShoppingMall",
    "Age",
    "CryoSleep_True",
    "HomePlanet_Earth",
    "HomePlanet_Europa",
    "VIP_True",
    "HomePlanet_Mars",
    "Destination_PSO J318.5-22",
    "VIP_False",
    "Destination_55 Cancri e",
    "FoodCourt",
    "Destination_TRAPPIST-1e",
]


def find_data_dir(explicit: Path | None = None) -> Path:
    """Find train/test/sample CSVs in a portable demo layout."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            Path.cwd(),
            Path.cwd() / "data",
            Path.cwd().parent,
            Path.cwd().parent / "data",
            PROJECT_ROOT,
            PROJECT_ROOT / "data",
        ]
    )
    for path in candidates:
        if all((path / name).exists() for name in ("train.csv", "test.csv", "sample_submission.csv")):
            return path
    checked = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find train.csv, test.csv, and sample_submission.csv. Checked:\n{checked}")


def make_one_hot_encoder() -> OneHotEncoder:
    """Keep the original sparse=False intent while supporting modern sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_score(model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Original notebook scoring helper: accuracy with cv=10."""
    return cross_val_score(model, X, y, scoring="accuracy", cv=10)


def prepare_original_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Apply the same feature engineering cells used by the original notebook."""
    train_test = train._append(test, ignore_index=True)

    expenses_columns = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    train_test.loc[:, expenses_columns] = train_test.apply(lambda x: 0 if x.CryoSleep == True else x, axis=1)
    train_test["Expenses"] = train_test.loc[:, expenses_columns].sum(axis=1)
    train_test.loc[:, ["CryoSleep"]] = train_test.apply(
        lambda x: True if x.Expenses == 0 and pd.isna(x.CryoSleep) else x,
        axis=1,
    )

    train_test.Name = train_test.Name.fillna("Unknown Unknown")
    train_test.loc[:, ["Room"]] = train_test.PassengerId.apply(lambda x: x[0:4])

    guide_vip = train_test.loc[:, ["Room", "VIP"]].dropna().drop_duplicates("Room")
    guide_cabin = train_test.loc[:, ["Room", "Cabin"]].dropna().drop_duplicates("Room")
    guide_homeplanet = train_test.loc[:, ["Room", "HomePlanet"]].dropna().drop_duplicates("Room")
    guide_destination = train_test.loc[:, ["Room", "Destination"]].dropna().drop_duplicates("Room")
    train_test = pd.merge(train_test, guide_cabin, how="left", on="Room", suffixes=("", "_y"))
    train_test = pd.merge(train_test, guide_vip, how="left", on="Room", suffixes=("", "_y"))
    train_test = pd.merge(train_test, guide_homeplanet, how="left", on="Room", suffixes=("", "_y"))
    train_test = pd.merge(train_test, guide_destination, how="left", on="Room", suffixes=("", "_y"))

    train_test.loc[:, ["VIP"]] = train_test.apply(lambda x: x.VIP_y if pd.isna(x.VIP) else x, axis=1)
    train_test.loc[:, ["Cabin"]] = train_test.apply(lambda x: x.Cabin_y if pd.isna(x.Cabin) else x, axis=1)
    train_test.loc[:, ["HomePlanet"]] = train_test.apply(
        lambda x: x.HomePlanet_y if pd.isna(x.HomePlanet) else x,
        axis=1,
    )
    train_test.loc[:, ["Destination"]] = train_test.apply(
        lambda x: x.Destination_y if pd.isna(x.Destination) else x,
        axis=1,
    )

    train_test.loc[:, ["Cabin_1"]] = train_test.Cabin.str.split("/", expand=True).iloc[:, 0]
    train_test.loc[:, ["Cabin_2"]] = train_test.Cabin.str.split("/", expand=True).iloc[:, 1]
    train_test.loc[:, ["Cabin_3"]] = train_test.Cabin.str.split("/", expand=True).iloc[:, 2]

    train_test.loc[:, ["FirstName"]] = train_test.Name.str.split(" ", expand=True).iloc[:, 0]
    train_test.loc[:, ["SecondName"]] = train_test.Name.str.split(" ", expand=True).iloc[:, 1]
    train_test["Name_key"] = train_test["SecondName"] + train_test["Room"]

    num_cols = ["ShoppingMall", "FoodCourt", "RoomService", "Spa", "VRDeck", "Expenses", "Age"]
    cat_cols = ["CryoSleep", "Cabin_1", "Cabin_3", "VIP", "HomePlanet", "Destination"]
    transported = ["Transported"]
    train_test = train_test[num_cols + cat_cols + transported].copy()

    num_imp = SimpleImputer(strategy="mean")
    cat_imp = SimpleImputer(strategy="most_frequent")
    ohe = make_one_hot_encoder()

    train_test[num_cols] = pd.DataFrame(num_imp.fit_transform(train_test[num_cols]), columns=num_cols)
    train_test[cat_cols] = pd.DataFrame(cat_imp.fit_transform(train_test[cat_cols]), columns=cat_cols)
    temp_train = pd.DataFrame(ohe.fit_transform(train_test[cat_cols]), columns=ohe.get_feature_names_out())
    train_test = train_test.drop(cat_cols, axis=1)
    train_test = pd.concat([train_test, temp_train], axis=1)

    train = train_test[train_test["Transported"].notnull()].copy()
    train.Transported = train.Transported.astype("int")
    test = train_test[train_test["Transported"].isnull()].drop("Transported", axis=1)

    X = train.drop("Transported", axis=1)
    y = train.Transported
    return X, y, test


def compare_reference(output: Path, reference: Path | None) -> dict[str, int | float | str | None]:
    if reference is None or not reference.exists():
        return {"reference_path": str(reference) if reference else None, "reference_available": 0}

    generated = pd.read_csv(output)
    expected = pd.read_csv(reference)
    if len(generated) != len(expected):
        return {
            "reference_path": str(reference),
            "reference_available": 1,
            "row_count_match": 0,
            "generated_rows": int(len(generated)),
            "reference_rows": int(len(expected)),
        }

    diff_count = int((generated["Transported"].astype(bool) != expected["Transported"].astype(bool)).sum())
    return {
        "reference_path": str(reference),
        "reference_available": 1,
        "row_count_match": 1,
        "row_diff_count": diff_count,
        "reference_true_count": int(expected["Transported"].astype(bool).sum()),
        "generated_true_count": int(generated["Transported"].astype(bool).sum()),
        "reference_true_rate": float(expected["Transported"].astype(bool).mean()),
        "generated_true_rate": float(generated["Transported"].astype(bool).mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strictly reproduce the original 0.814 Optuna XGB notebook branch.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing train.csv/test.csv/sample_submission.csv.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "submissions" / "Submission_XGB_exact.csv")
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "tables" / "notebook_0814_exact_run_metrics.json",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=PROJECT_ROOT / "submissions" / "submission_xgb_reference_0814.csv",
        help="Optional highest-score CSV used only for provenance comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = find_data_dir(args.data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

    X, y, test_matrix = prepare_original_features(train, test)
    X, y = shuffle(X, y)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    original_cv_mean = float(get_score(xgb.XGBClassifier(**PARAMS_XGB_BEST), X, y).mean())

    features_isolation = ["ShoppingMall", "FoodCourt", "RoomService", "Spa", "VRDeck", "Age"]
    isf = IsolationForest(n_jobs=-1, random_state=1, n_estimators=100, contamination=0.003)
    isf.fit(X[features_isolation], y)
    rows = pd.DataFrame(isf.predict(X[features_isolation]), columns=["feature"])
    rows_ind = rows[rows.feature == 1]
    X_1 = X.iloc[rows_ind.index].reset_index(drop=True)
    y_1 = y.iloc[rows_ind.index].reset_index(drop=True)
    isolation_cv_mean = float(get_score(xgb.XGBClassifier(**PARAMS_XGB_BEST), X_1, y_1).mean())

    X = X.drop(DROP_LIST, axis=1)
    test_matrix = test_matrix.drop(DROP_LIST, axis=1)
    pruned_cv_mean = float(get_score(xgb.XGBClassifier(**PARAMS_XGB_BEST), X, y).mean())

    sm = KMeansSMOTE(sampling_strategy=1, n_jobs=-1)
    X_sm, y_sm = sm.fit_resample(X, y)
    X = X_sm
    y = y_sm

    pred_xgb = xgb.XGBClassifier(**PARAMS_XGB_BEST).fit(X, y).predict(test_matrix)
    submission = sample_submission.copy()
    submission["Transported"] = pred_xgb
    submission["Transported"] = submission["Transported"] > 0.5

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)

    metrics = {
        "source_notebook": "0-814-optuna-xgb-space-titanic.ipynb",
        "note": "Original notebook leaves shuffle, KMeansSMOTE, and XGB estimator randomness unseeded.",
        "original_cv_mean_after_shuffle": original_cv_mean,
        "isolation_diagnostic_cv_mean": isolation_cv_mean,
        "pruned_cv_mean": pruned_cv_mean,
        "features_before_drop": int(len(DROP_LIST) + X.shape[1]),
        "features_after_drop": int(test_matrix.shape[1]),
        "smote_class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().to_dict().items()},
        "output_true_count": int(submission["Transported"].sum()),
        "output_true_rate": float(submission["Transported"].mean()),
        "params_xgb_best": PARAMS_XGB_BEST,
        "reference_comparison": compare_reference(args.output, args.reference),
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Data directory: {data_dir}")
    print(f"Saved submission: {args.output}")
    print(f"Saved metrics: {args.metrics_output}")
    print(f"CV after original shuffle cell: {original_cv_mean:.6f}")
    print(f"CV for IsolationForest diagnostic branch: {isolation_cv_mean:.6f}")
    print(f"CV after original drop_list cell: {pruned_cv_mean:.6f}")
    print(f"SMOTE class counts: {metrics['smote_class_counts']}")
    print(f"Output true count/rate: {metrics['output_true_count']} / {metrics['output_true_rate']:.6f}")
    ref = metrics["reference_comparison"]
    if ref.get("reference_available"):
        print(f"Reference row diff count: {ref.get('row_diff_count')}")


if __name__ == "__main__":
    main()
