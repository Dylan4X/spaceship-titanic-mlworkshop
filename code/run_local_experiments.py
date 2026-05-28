"""Run compact local benchmark and ablation experiments.

The Kaggle notebook is the source for the final 0.81716 submission. This file
keeps a small, readable local experiment runner for the report evidence: model
comparison plus feature-family ablation under the same prepared feature matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from reproduce_081716_xgb import SEED, XGB_PARAMS, find_data_dir, prepare_frames  # noqa: E402


def cv_accuracy(model, x: pd.DataFrame, y: pd.Series, folds: int = 5) -> tuple[float, float]:
    scores: list[float] = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for train_idx, valid_idx in splitter.split(x, y):
        x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model.fit(x_train, y_train)
        scores.append(accuracy_score(y_valid, model.predict(x_valid)))
    return float(np.mean(scores)), float(np.std(scores))


def model_table(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=SEED,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=600,
            learning_rate=0.035,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEED,
            verbose=-1,
        ),
        "XGBoost": XGBClassifier(**XGB_PARAMS, random_state=SEED, eval_metric="logloss"),
        "CatBoost": CatBoostClassifier(
            iterations=700,
            learning_rate=0.035,
            depth=6,
            loss_function="Logloss",
            random_seed=SEED,
            verbose=False,
        ),
    }

    rows = []
    for name, model in models.items():
        mean, std = cv_accuracy(model, x, y)
        rows.append({"experiment": name, "cv_accuracy_mean": mean, "cv_accuracy_std": std})
        print(f"{name}: {mean:.5f} +/- {std:.5f}")
    return pd.DataFrame(rows)


def drop_matching(x: pd.DataFrame, tokens: tuple[str, ...]) -> pd.DataFrame:
    cols = [col for col in x.columns if not any(token in col for token in tokens)]
    return x.loc[:, cols]


def ablation_table(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    groups = {
        "all_features": (),
        "without_expense_features": ("RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Expenses"),
        "without_cabin_features": ("Cabin_1", "Cabin_3"),
        "without_route_features": ("HomePlanet", "Destination"),
        "without_cryo_vip_features": ("CryoSleep", "VIP"),
    }
    rows = []
    for name, tokens in groups.items():
        x_part = drop_matching(x, tokens)
        model = XGBClassifier(**XGB_PARAMS, random_state=SEED, eval_metric="logloss")
        mean, std = cv_accuracy(model, x_part, y)
        rows.append(
            {
                "experiment": name,
                "n_features": x_part.shape[1],
                "cv_accuracy_mean": mean,
                "cv_accuracy_std": std,
            }
        )
        print(f"{name}: {mean:.5f} +/- {std:.5f} ({x_part.shape[1]} features)")
    return pd.DataFrame(rows)


def main() -> None:
    data_dir = find_data_dir()
    train_frame, _ = prepare_frames(data_dir)
    x = train_frame.drop(columns=["Transported"])
    y = train_frame["Transported"].astype(int)

    output_dir = PROJECT_ROOT / "experiments" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = model_table(x, y)
    benchmark.to_csv(output_dir / "local_cv_benchmark.csv", index=False)

    ablation = ablation_table(x, y)
    ablation.to_csv(output_dir / "local_feature_ablation.csv", index=False)

    print("Saved:", output_dir / "local_cv_benchmark.csv")
    print("Saved:", output_dir / "local_feature_ablation.csv")


if __name__ == "__main__":
    main()
