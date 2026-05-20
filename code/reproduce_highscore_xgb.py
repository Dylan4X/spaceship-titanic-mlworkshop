"""Cleaned companion rerun for the compact XGBoost branch.

This is useful for the supporting Optuna/neighborhood evidence. For the strict
teacher-facing reproduction of `0-814-optuna-xgb-space-titanic.ipynb`, use
`code/reproduce_0_814_notebook_exact.py` and
`notebooks/0_814_exact_reproduction_demo.ipynb`.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import KMeansSMOTE
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

XGB_BEST_PARAMS = {
    "reg_lambda": 3.06,
    "reg_alpha": 4.582,
    "colsample_bytree": 0.93,
    "subsample": 0.96,
    "n_estimators": 725,
    "max_depth": 5,
    "learning_rate": 0.05,
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
    "eval_metric": "logloss",
    "verbosity": 0,
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
    """Find data whether CSVs are in ./data or directly beside the project files."""
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    script_root = Path(__file__).resolve().parents[1]
    candidates.extend([Path.cwd() / "data", Path.cwd(), script_root / "data", script_root])
    for path in candidates:
        if (path / "train.csv").exists() and (path / "test.csv").exists() and (path / "sample_submission.csv").exists():
            return path
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find train.csv, test.csv, and sample_submission.csv. Checked: {checked}")


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_compact_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    all_df = pd.concat([train_df, test_df], ignore_index=True)

    all_df.loc[all_df["CryoSleep"].eq(True), SPEND_COLS] = 0
    all_df["Expenses"] = all_df[SPEND_COLS].sum(axis=1)
    all_df.loc[(all_df["Expenses"].eq(0)) & all_df["CryoSleep"].isna(), "CryoSleep"] = True

    all_df["Name"] = all_df["Name"].fillna("Unknown Unknown")
    all_df["Room"] = all_df["PassengerId"].astype(str).str.slice(0, 4)

    for col in ["VIP", "Cabin", "HomePlanet", "Destination"]:
        guide = all_df[["Room", col]].dropna().drop_duplicates("Room").set_index("Room")[col]
        all_df[col] = all_df[col].fillna(all_df["Room"].map(guide))

    cabin_split = all_df["Cabin"].str.split("/", expand=True)
    all_df["Cabin_1"] = cabin_split[0]
    all_df["Cabin_2"] = cabin_split[1]
    all_df["Cabin_3"] = cabin_split[2]

    name_split = all_df["Name"].str.split(" ", expand=True)
    all_df["FirstName"] = name_split[0]
    all_df["SecondName"] = name_split[1]
    all_df["Name_key"] = all_df["SecondName"] + all_df["Room"]

    num_cols = ["ShoppingMall", "FoodCourt", "RoomService", "Spa", "VRDeck", "Expenses", "Age"]
    cat_cols = ["CryoSleep", "Cabin_1", "Cabin_3", "VIP", "HomePlanet", "Destination"]
    compact = all_df[num_cols + cat_cols + ["Transported"]].copy()

    compact[num_cols] = pd.DataFrame(
        SimpleImputer(strategy="mean").fit_transform(compact[num_cols]),
        columns=num_cols,
    )
    compact[cat_cols] = pd.DataFrame(
        SimpleImputer(strategy="most_frequent").fit_transform(compact[cat_cols]),
        columns=cat_cols,
    )

    encoder = make_one_hot_encoder()
    encoded = pd.DataFrame(encoder.fit_transform(compact[cat_cols]), columns=encoder.get_feature_names_out())
    compact = pd.concat(
        [compact.drop(columns=cat_cols).reset_index(drop=True), encoded.reset_index(drop=True)],
        axis=1,
    )

    train_proc = compact[compact["Transported"].notna()].copy()
    train_proc["Transported"] = train_proc["Transported"].astype(int)
    test_proc = compact[compact["Transported"].isna()].drop(columns="Transported").copy()

    X = train_proc.drop(columns="Transported")
    y = train_proc["Transported"]
    groups = train_df["PassengerId"].astype(str).str.split("_").str[0].reset_index(drop=True)
    return X, y, test_proc, groups


def cv_accuracy(params: dict, X: pd.DataFrame, y: pd.Series, n_splits: int) -> tuple[float, float]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(XGBClassifier(**params), X, y, scoring="accuracy", cv=splitter)
    return float(scores.mean()), float(scores.std())


def run_param_check(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> pd.DataFrame:
    candidates = {
        "final_params": {},
        "500_trees": {"n_estimators": 500},
        "900_trees": {"n_estimators": 900},
        "depth_4": {"max_depth": 4},
        "depth_6": {"max_depth": 6},
        "lr_0.06": {"learning_rate": 0.06, "n_estimators": 600},
    }

    rows = []
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for name, override in candidates.items():
        params = {**XGB_BEST_PARAMS, **override}
        fold_scores = []
        for train_idx, valid_idx in splitter.split(X, y, groups=groups):
            model = XGBClassifier(**params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[valid_idx])
            fold_scores.append(accuracy_score(y.iloc[valid_idx], pred))
        rows.append(
            {
                "candidate": name,
                "mean_accuracy": float(np.mean(fold_scores)),
                "std_accuracy": float(np.std(fold_scores)),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_accuracy", ascending=False).reset_index(drop=True)


def run_optuna_search(X: pd.DataFrame, y: pd.Series, n_trials: int) -> tuple[dict, float]:
    def objective(trial: optuna.Trial) -> float:
        params = {
            **XGB_BEST_PARAMS,
            "n_estimators": trial.suggest_int("n_estimators", 400, 900, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.08, step=0.01),
            "subsample": trial.suggest_float("subsample", 0.85, 1.0, step=0.05),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.80, 1.0, step=0.05),
            "reg_alpha": trial.suggest_float("reg_alpha", 2.0, 7.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0),
        }
        mean_acc, _ = cv_accuracy(params, X, y, n_splits=3)
        return mean_acc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, float(study.best_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the high-score XGBoost branch.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing train/test/sample CSVs.")
    parser.add_argument("--output", type=Path, default=Path("submissions/Submission_XGB_companion.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("experiments/tables/xgb_demo_run_metrics.json"))
    parser.add_argument("--n-trials", type=int, default=8, help="Small Optuna search size for the demo run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = find_data_dir(args.data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

    X_full, y, test_full, groups = build_compact_features(train_df, test_df)
    base_mean, base_std = cv_accuracy(XGB_BEST_PARAMS, X_full, y, n_splits=3)

    features_isolation = ["ShoppingMall", "FoodCourt", "RoomService", "Spa", "VRDeck", "Age"]
    isolation = IsolationForest(n_jobs=-1, random_state=1, n_estimators=100, contamination=0.003)
    keep_mask = isolation.fit_predict(X_full[features_isolation]) == 1

    X_model = X_full.drop(columns=DROP_LIST)
    test_model = test_full.drop(columns=DROP_LIST)

    param_check = run_param_check(X_model, y, groups)
    optuna_best_params, optuna_best_value = run_optuna_search(X_model, y, args.n_trials)

    smote = KMeansSMOTE(sampling_strategy=1, n_jobs=-1, random_state=RANDOM_STATE)
    X_train_final, y_train_final = smote.fit_resample(X_model, y)

    final_model = XGBClassifier(**XGB_BEST_PARAMS)
    final_model.fit(X_train_final, y_train_final)
    test_pred = final_model.predict(test_model)

    submission = sample_submission.copy()
    submission["Transported"] = test_pred.astype(bool)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)

    metrics = {
        "processed_feature_count": int(X_full.shape[1]),
        "pruned_feature_count": int(X_model.shape[1]),
        "isolation_forest_retained_rows": int(keep_mask.sum()),
        "base_3fold_accuracy_mean": base_mean,
        "base_3fold_accuracy_std": base_std,
        "kmeans_smote_rows": int(X_train_final.shape[0]),
        "predicted_true_count": int(submission["Transported"].sum()),
        "predicted_true_rate": float(submission["Transported"].mean()),
        "optuna_n_trials": int(args.n_trials),
        "optuna_best_value": optuna_best_value,
        "optuna_best_params": optuna_best_params,
        "param_check": param_check.to_dict(orient="records"),
        "final_fixed_params": XGB_BEST_PARAMS,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved submission: {args.output}")
    print(f"Saved metrics: {args.metrics_output}")
    print(f"True count/rate: {metrics['predicted_true_count']} / {metrics['predicted_true_rate']:.4f}")
    print(f"Optuna best 3-fold CV ({args.n_trials} trials): {optuna_best_value:.5f}")


if __name__ == "__main__":
    main()
