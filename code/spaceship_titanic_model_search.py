import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from project_paths import DATA_DIR, EXPERIMENTS_TABLE_DIR


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train.csv"
RESULTS_PATH = EXPERIMENTS_TABLE_DIR / "model_search_results.json"

N_SPLITS = 5
SEARCH_SEEDS = [7, 21, 42, 87, 123]
AUDIT_SEEDS = [3, 11, 17, 29, 57, 68, 99, 131]
MIN_PROMOTION_DELTA = 0.001
BLEND_WEIGHT_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

FEATURE_COLUMNS = [
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

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [col for col in FEATURE_COLUMNS if col not in NUMERIC_FEATURES]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
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

    df["CryoSleep"] = df["CryoSleep"].astype("object")
    df["VIP"] = df["VIP"].astype("object")
    df["HomePlanet"] = df["HomePlanet"].astype("object")
    df["Destination"] = df["Destination"].astype("object")

    df["AgeBand"] = pd.cut(
        df["Age"],
        bins=[-1, 12, 18, 25, 40, 60, 200],
        labels=["child", "teen", "young_adult", "adult", "middle_age", "senior"],
    ).astype("object")

    df["CryoNoSpendMatch"] = (
        df["CryoSleep"].fillna("Unknown").eq("True") & (df["NoSpend"] == 1)
    ).astype(int)
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
    df["NonZeroSpendCount"] = (df[spend_cols].fillna(0) > 0).sum(axis=1)
    df["LogTotalSpend"] = np.log1p(df["TotalSpend"])
    df["CabinNumBin"] = pd.cut(
        df["CabinNum"],
        bins=[-1, 300, 600, 900, 1200, 2000],
        labels=["front", "mid1", "mid2", "rear", "far"],
    ).astype("object")
    df["YoungCryo"] = ((df["Age"].fillna(30) < 18) & df["CryoSleep"].fillna("Unknown").eq("True")).astype(int)

    return df


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def make_lgb_a(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
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


def make_xgb_a(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
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


def evaluate_single_model(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, seeds: list[int], pipeline_factory
) -> dict:
    fold_scores = []
    oof_prob_sum = np.zeros(len(X), dtype=float)
    oof_counts = np.zeros(len(X), dtype=int)

    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for train_idx, valid_idx in splitter.split(X, y, groups):
            pipeline = pipeline_factory(seed)
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            valid_prob = pipeline.predict_proba(X.iloc[valid_idx])[:, 1]
            valid_pred = valid_prob >= 0.5
            fold_scores.append(float(accuracy_score(y.iloc[valid_idx], valid_pred)))
            oof_prob_sum[valid_idx] += valid_prob
            oof_counts[valid_idx] += 1

    oof_prob = oof_prob_sum / oof_counts
    return {
        "fold_mean": float(np.mean(fold_scores)),
        "fold_std": float(np.std(fold_scores)),
        "oof_accuracy_at_05": float(accuracy_score(y, oof_prob >= 0.5)),
        "oof_probabilities": oof_prob.tolist(),
    }


def evaluate_blend(y: pd.Series, model_outputs: dict, weights: dict[str, float]) -> dict:
    blend = np.zeros(len(y), dtype=float)
    for name, weight in weights.items():
        blend += weight * np.array(model_outputs[name])
    blend /= sum(weights.values())
    return {
        "oof_accuracy_at_05": float(accuracy_score(y, blend >= 0.5)),
        "weights": weights,
    }


def main() -> None:
    train_df = add_features(pd.read_csv(TRAIN_PATH))
    y = pd.read_csv(TRAIN_PATH)["Transported"].astype(int)
    groups = train_df["GroupId"]
    X = train_df[FEATURE_COLUMNS]

    search_outputs = {}
    audit_outputs = {}

    single_models = {
        "lgb_a": make_lgb_a,
        "xgb_a": make_xgb_a,
    }

    for name, factory in single_models.items():
        search_outputs[name] = evaluate_single_model(X, y, groups, SEARCH_SEEDS, factory)
        audit_outputs[name] = evaluate_single_model(X, y, groups, AUDIT_SEEDS, factory)

    summary = []
    baseline_search = search_outputs["lgb_a"]["oof_accuracy_at_05"]
    baseline_audit = audit_outputs["lgb_a"]["oof_accuracy_at_05"]

    for name in single_models:
        summary.append(
            {
                "model": name,
                "search_fold_mean": search_outputs[name]["fold_mean"],
                "search_oof_accuracy_at_05": search_outputs[name]["oof_accuracy_at_05"],
                "audit_fold_mean": audit_outputs[name]["fold_mean"],
                "audit_oof_accuracy_at_05": audit_outputs[name]["oof_accuracy_at_05"],
                "promote": False if name == "lgb_a" else (
                    search_outputs[name]["oof_accuracy_at_05"] >= baseline_search + MIN_PROMOTION_DELTA
                    and audit_outputs[name]["oof_accuracy_at_05"] >= baseline_audit + MIN_PROMOTION_DELTA
                ),
            }
        )

    blend_results = []
    for lgb_weight in BLEND_WEIGHT_GRID:
        weights = {"lgb_a": lgb_weight, "xgb_a": 1.0 - lgb_weight}
        blend_search = evaluate_blend(
            y,
            {name: result["oof_probabilities"] for name, result in search_outputs.items()},
            weights,
        )
        blend_audit = evaluate_blend(
            y,
            {name: result["oof_probabilities"] for name, result in audit_outputs.items()},
            weights,
        )
        promote = (
            blend_search["oof_accuracy_at_05"] >= baseline_search + MIN_PROMOTION_DELTA
            and blend_audit["oof_accuracy_at_05"] >= baseline_audit + MIN_PROMOTION_DELTA
        )
        blend_results.append(
            {
                "model": f"blend_lgb{lgb_weight:.2f}_xgb{1.0 - lgb_weight:.2f}",
                "search_fold_mean": None,
                "search_oof_accuracy_at_05": blend_search["oof_accuracy_at_05"],
                "audit_fold_mean": None,
                "audit_oof_accuracy_at_05": blend_audit["oof_accuracy_at_05"],
                "promote": promote,
                "weights": weights,
            }
        )

    blend_results.sort(
        key=lambda item: (item["promote"], item["audit_oof_accuracy_at_05"], item["search_oof_accuracy_at_05"]),
        reverse=True,
    )
    summary.extend(blend_results)

    payload = {
        "protocol": {
            "cv_scheme": "Repeated StratifiedGroupKFold",
            "n_splits": N_SPLITS,
            "search_seeds": SEARCH_SEEDS,
            "audit_seeds": AUDIT_SEEDS,
            "promotion_rule": (
                f"Promote only if both search and audit OOF improve by at least {MIN_PROMOTION_DELTA:.4f} "
                "over lgb_a at threshold 0.5."
            ),
            "notes": [
                "Do not tune threshold on audit seeds.",
                "Do not accept 0.000x gains as real improvements.",
                "Use search seeds for exploration, audit seeds for promotion decisions.",
            ],
        },
        "results": summary,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
