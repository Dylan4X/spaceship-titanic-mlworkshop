"""Reproduce the local validation experiments used in the report.

This module consolidates the local model-search code into a portable script.
It reproduces the validation-first line: engineered features, repeated
StratifiedGroupKFold, search/audit seed separation, LightGBM/XGBoost comparison,
the promoted blend check, and the retained LightGBM tuning evidence.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", message="X does not have valid feature names")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TABLE_DIR = PROJECT_ROOT / "experiments" / "tables"
OUTPUT_JSON = TABLE_DIR / "local_validation_reproduction.json"
OUTPUT_CSV = TABLE_DIR / "local_validation_reproduction.csv"

N_SPLITS = 5
SEARCH_SEEDS = [7, 21, 42, 87, 123]
AUDIT_SEEDS = [3, 11, 17, 29, 57, 68, 99, 131]
MIN_PROMOTION_DELTA = 0.001

MODEL_SEARCH_FEATURES = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "Age",
    "VIP",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "GroupMember",
    "GroupSize",
    "IsAlone",
    "CabinDeck",
    "CabinNum",
    "CabinSide",
    "TotalSpend",
    "NoSpend",
    "LuxurySpend",
    "BasicSpend",
    "AgeBand",
    "CryoNoSpendMatch",
    "AgeMissing",
    "CabinKnown",
    "CabinDeckSide",
    "Route",
    "SpendBins",
    "NonZeroSpendCount",
    "LogTotalSpend",
    "CabinNumBin",
    "YoungCryo",
]

MODEL_SEARCH_NUMERIC = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "GroupMember",
    "GroupSize",
    "IsAlone",
    "CabinNum",
    "TotalSpend",
    "NoSpend",
    "LuxurySpend",
    "BasicSpend",
    "CryoNoSpendMatch",
    "AgeMissing",
    "CabinKnown",
    "NonZeroSpendCount",
    "LogTotalSpend",
    "YoungCryo",
]

MODEL_SEARCH_CATEGORICAL = [col for col in MODEL_SEARCH_FEATURES if col not in MODEL_SEARCH_NUMERIC]

SIMPLE_FEATURES = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "Age",
    "VIP",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "GroupMember",
    "GroupSize",
    "IsAlone",
    "CabinDeck",
    "CabinNum",
    "CabinSide",
    "TotalSpend",
    "NoSpend",
    "LuxurySpend",
    "BasicSpend",
    "AgeBand",
    "CryoNoSpendMatch",
]

SIMPLE_NUMERIC = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "GroupMember",
    "GroupSize",
    "IsAlone",
    "CabinNum",
    "TotalSpend",
    "NoSpend",
    "LuxurySpend",
    "BasicSpend",
    "CryoNoSpendMatch",
]

SIMPLE_CATEGORICAL = [col for col in SIMPLE_FEATURES if col not in SIMPLE_NUMERIC]

LGB_DEFAULT_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "min_child_samples": 25,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}

LGB_TUNED_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 6,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_samples": 35,
    "reg_alpha": 0.7,
    "reg_lambda": 0.1,
}


def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "train.csv")


def add_model_search_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_simple_features(df)

    df["AgeMissing"] = df["Age"].isna().astype(int)
    df["CabinKnown"] = df["Cabin"].notna().astype(int)
    df["CabinDeckSide"] = (
        df["CabinDeck"].fillna("Unknown").astype(str) + "_" + df["CabinSide"].fillna("Unknown").astype(str)
    ).astype("object")
    df["Route"] = (
        df["HomePlanet"].fillna("Unknown").astype(str) + "_" + df["Destination"].fillna("Unknown").astype(str)
    ).astype("object")
    df["SpendBins"] = pd.cut(
        df["TotalSpend"],
        bins=[-1, 0, 1, 100, 1000, 5000, 1e9],
        labels=["0", "1", "low", "mid", "high", "very_high"],
    ).astype("object")
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["NonZeroSpendCount"] = (df[spend_cols].fillna(0) > 0).sum(axis=1)
    df["LogTotalSpend"] = np.log1p(df["TotalSpend"])
    df["CabinNumBin"] = pd.cut(
        df["CabinNum"],
        bins=[-1, 300, 600, 900, 1200, 2000],
        labels=["front", "mid1", "mid2", "rear", "far"],
    ).astype("object")
    df["YoungCryo"] = ((df["Age"].fillna(30) < 18) & df["CryoSleep"].fillna("Unknown").eq("True")).astype(int)
    return df


def add_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    group_split = df["PassengerId"].str.split("_", expand=True)
    df["GroupId"] = group_split[0]
    df["GroupMember"] = pd.to_numeric(group_split[1], errors="coerce")
    df["GroupSize"] = df.groupby("GroupId")["PassengerId"].transform("count")
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)

    cabin_split = df["Cabin"].fillna("Unknown/Unknown/Unknown").str.split("/", expand=True)
    df["CabinDeck"] = cabin_split[0]
    df["CabinNum"] = pd.to_numeric(cabin_split[1].replace("Unknown", np.nan), errors="coerce")
    df["CabinSide"] = cabin_split[2]

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in spend_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["TotalSpend"] = df[spend_cols].fillna(0).sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["LuxurySpend"] = df[["Spa", "VRDeck", "FoodCourt"]].fillna(0).sum(axis=1)
    df["BasicSpend"] = df[["RoomService", "ShoppingMall"]].fillna(0).sum(axis=1)

    for col in ["CryoSleep", "VIP", "HomePlanet", "Destination"]:
        df[col] = df[col].astype("object")
    df["AgeBand"] = pd.cut(
        df["Age"],
        bins=[-1, 12, 18, 25, 40, 60, 200],
        labels=["child", "teen", "young_adult", "adult", "middle_age", "senior"],
    ).astype("object")
    df["CryoNoSpendMatch"] = (
        df["CryoSleep"].fillna("Unknown").eq("True") & (df["NoSpend"] == 1)
    ).astype(int)
    return df


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def make_lgb_model_search(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(MODEL_SEARCH_NUMERIC, MODEL_SEARCH_CATEGORICAL)),
            (
                "model",
                LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.03,
                    num_leaves=31,
                    min_child_samples=25,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=seed,
                    objective="binary",
                    verbosity=-1,
                ),
            ),
        ]
    )


def make_xgb_model_search(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(MODEL_SEARCH_NUMERIC, MODEL_SEARCH_CATEGORICAL)),
            (
                "model",
                XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.03,
                    max_depth=5,
                    min_child_weight=2,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=seed,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                ),
            ),
        ]
    )


def make_simple_lgb(params: dict, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(SIMPLE_NUMERIC, SIMPLE_CATEGORICAL)),
            (
                "model",
                LGBMClassifier(
                    **params,
                    random_state=seed,
                    objective="binary",
                    verbosity=-1,
                ),
            ),
        ]
    )


def repeated_group_oof(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    seeds: list[int],
    pipeline_factory,
) -> dict:
    fold_scores: list[float] = []
    oof_prob_sum = np.zeros(len(x), dtype=float)
    oof_counts = np.zeros(len(x), dtype=int)

    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for train_idx, valid_idx in splitter.split(x, y, groups):
            pipeline = pipeline_factory(seed)
            pipeline.fit(x.iloc[train_idx], y.iloc[train_idx])
            valid_prob = pipeline.predict_proba(x.iloc[valid_idx])[:, 1]
            fold_scores.append(float(accuracy_score(y.iloc[valid_idx], valid_prob >= 0.5)))
            oof_prob_sum[valid_idx] += valid_prob
            oof_counts[valid_idx] += 1

    oof_prob = oof_prob_sum / oof_counts
    return {
        "fold_mean": float(np.mean(fold_scores)),
        "fold_std": float(np.std(fold_scores)),
        "oof_accuracy_at_05": float(accuracy_score(y, oof_prob >= 0.5)),
        "oof_probabilities": oof_prob,
    }


def blend_accuracy(y: pd.Series, outputs: dict[str, np.ndarray], lgb_weight: float) -> float:
    blend = lgb_weight * outputs["lgb_a"] + (1.0 - lgb_weight) * outputs["xgb_a"]
    return float(accuracy_score(y, blend >= 0.5))


def reproduce_model_search() -> list[dict]:
    train = add_model_search_features(load_train())
    x = train[MODEL_SEARCH_FEATURES]
    y = train["Transported"].astype(int)
    groups = train["GroupId"]

    factories = {
        "lgb_a": make_lgb_model_search,
        "xgb_a": make_xgb_model_search,
    }
    search_outputs = {
        name: repeated_group_oof(x, y, groups, SEARCH_SEEDS, factory) for name, factory in factories.items()
    }
    audit_outputs = {
        name: repeated_group_oof(x, y, groups, AUDIT_SEEDS, factory) for name, factory in factories.items()
    }

    rows = []
    baseline_search = search_outputs["lgb_a"]["oof_accuracy_at_05"]
    baseline_audit = audit_outputs["lgb_a"]["oof_accuracy_at_05"]

    for name in factories:
        rows.append(
            {
                "experiment_group": "model_search",
                "model": name,
                "search_oof_accuracy_at_05": search_outputs[name]["oof_accuracy_at_05"],
                "audit_oof_accuracy_at_05": audit_outputs[name]["oof_accuracy_at_05"],
                "search_fold_mean": search_outputs[name]["fold_mean"],
                "audit_fold_mean": audit_outputs[name]["fold_mean"],
                "promote": False
                if name == "lgb_a"
                else (
                    search_outputs[name]["oof_accuracy_at_05"] >= baseline_search + MIN_PROMOTION_DELTA
                    and audit_outputs[name]["oof_accuracy_at_05"] >= baseline_audit + MIN_PROMOTION_DELTA
                ),
                "notes": "Original validation-first LGB/XGB comparison",
            }
        )

    search_probs = {name: result["oof_probabilities"] for name, result in search_outputs.items()}
    audit_probs = {name: result["oof_probabilities"] for name, result in audit_outputs.items()}
    blend_search = blend_accuracy(y, search_probs, 0.70)
    blend_audit = blend_accuracy(y, audit_probs, 0.70)
    rows.append(
        {
            "experiment_group": "model_search",
            "model": "blend_lgb07_xgb03",
            "search_oof_accuracy_at_05": blend_search,
            "audit_oof_accuracy_at_05": blend_audit,
            "search_fold_mean": np.nan,
            "audit_fold_mean": np.nan,
            "promote": blend_search >= baseline_search + MIN_PROMOTION_DELTA
            and blend_audit >= baseline_audit + MIN_PROMOTION_DELTA,
            "notes": "Promoted blend from local experiment log",
        }
    )
    return rows


def reproduce_lgb_tuning() -> list[dict]:
    train = add_simple_features(load_train())
    x = train[SIMPLE_FEATURES]
    y = train["Transported"].astype(int)
    groups = train["GroupId"]

    rows = []
    for label, params in [("lgb_default", LGB_DEFAULT_PARAMS), ("lgb_tuned", LGB_TUNED_PARAMS)]:
        search = repeated_group_oof(x, y, groups, [7, 21, 42], lambda seed, p=params: make_simple_lgb(p, seed))
        audit = repeated_group_oof(x, y, groups, AUDIT_SEEDS, lambda seed, p=params: make_simple_lgb(p, seed))
        rows.append(
            {
                "experiment_group": "lgb_tuning",
                "model": label,
                "search_oof_accuracy_at_05": search["oof_accuracy_at_05"],
                "audit_oof_accuracy_at_05": audit["oof_accuracy_at_05"],
                "search_fold_mean": search["fold_mean"],
                "audit_fold_mean": audit["fold_mean"],
                "promote": label == "lgb_tuned",
                "notes": json.dumps(params, sort_keys=True),
            }
        )
    return rows


def run_reproduction() -> pd.DataFrame:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = reproduce_model_search() + reproduce_lgb_tuning()
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_CSV, index=False)
    json_rows = json.loads(results.to_json(orient="records"))
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "protocol": {
                    "cv_scheme": "Repeated StratifiedGroupKFold by PassengerId prefix",
                    "search_seeds": SEARCH_SEEDS,
                    "audit_seeds": AUDIT_SEEDS,
                    "promotion_delta": MIN_PROMOTION_DELTA,
                    "source": "Consolidated from the project local model-search scripts.",
                },
                "results": json_rows,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return results


def main() -> None:
    results = run_reproduction()
    print(results.to_string(index=False))
    print("Saved:", OUTPUT_CSV)
    print("Saved:", OUTPUT_JSON)


if __name__ == "__main__":
    main()
