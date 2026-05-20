import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from project_paths import DATA_DIR, EXPERIMENTS_TABLE_DIR, SUBMISSIONS_DIR


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
BEST_KNOWN_PATH = ROOT / "submission (3).csv"
RESULTS_PATH = EXPERIMENTS_TABLE_DIR / "simple_search_results.json"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_simple_tuned.csv"

N_SPLITS = 5
SEARCH_SEEDS = [7, 21, 42]
AUDIT_SEEDS = [3, 11, 17, 29, 57, 68, 99, 131]
RANDOM_STATE = 20260411

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
]

CATEGORICAL_FEATURES = [col for col in FEATURE_COLUMNS if col not in NUMERIC_FEATURES]

DEFAULT_PARAMS = {
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

PARAM_DISTRIBUTIONS = {
    "n_estimators": [350, 450, 500, 650, 800, 1000],
    "learning_rate": [0.015, 0.02, 0.025, 0.03, 0.04, 0.05],
    "num_leaves": [15, 23, 31, 47, 63],
    "max_depth": [-1, 4, 5, 6, 7],
    "subsample": [0.75, 0.85, 0.9, 0.95, 1.0],
    "colsample_bytree": [0.65, 0.75, 0.8, 0.9, 1.0],
    "min_child_samples": [10, 15, 20, 25, 35, 50, 75],
    "reg_alpha": [0.0, 0.01, 0.05, 0.1, 0.3, 0.7],
    "reg_lambda": [0.0, 0.05, 0.1, 0.3, 0.7, 1.5, 3.0],
}


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


def make_pipeline(params: dict, seed: int) -> Pipeline:
    preprocessor = ColumnTransformer(
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
    model = LGBMClassifier(
        **params,
        random_state=seed,
        objective="binary",
        verbosity=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def repeated_group_oof(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, params: dict, seeds: list[int]
) -> tuple[float, float, float]:
    fold_scores = []
    oof_prob_sum = np.zeros(len(X), dtype=float)
    oof_counts = np.zeros(len(X), dtype=int)

    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for train_idx, valid_idx in splitter.split(X, y, groups):
            pipeline = make_pipeline(params, seed)
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            valid_prob = pipeline.predict_proba(X.iloc[valid_idx])[:, 1]
            fold_scores.append(float(accuracy_score(y.iloc[valid_idx], valid_prob >= 0.5)))
            oof_prob_sum[valid_idx] += valid_prob
            oof_counts[valid_idx] += 1

    oof_prob = oof_prob_sum / oof_counts
    return (
        float(np.mean(fold_scores)),
        float(np.std(fold_scores)),
        float(accuracy_score(y, oof_prob >= 0.5)),
    )


def full_fit_submission(
    train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict, seed: int = 42
) -> pd.DataFrame:
    pipeline = make_pipeline(params, seed)
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df["Transported"].astype(int))
    test_prob = pipeline.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    return pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"],
            "Transported": test_prob >= 0.5,
        }
    )


def compare_to_best_known(submission: pd.DataFrame) -> dict:
    if not BEST_KNOWN_PATH.exists():
        return {}
    best_known = pd.read_csv(BEST_KNOWN_PATH)
    diff = submission["Transported"].ne(best_known["Transported"])
    return {
        "best_known_path": str(BEST_KNOWN_PATH),
        "diff_count": int(diff.sum()),
        "diff_rate": float(diff.mean()),
        "best_known_true_rate": float(best_known["Transported"].mean()),
    }


def main() -> None:
    train_df = add_features(pd.read_csv(TRAIN_PATH))
    test_df = add_features(pd.read_csv(TEST_PATH))
    X = train_df[FEATURE_COLUMNS]
    y = train_df["Transported"].astype(int)
    groups = train_df["GroupId"]

    baseline_search = repeated_group_oof(X, y, groups, DEFAULT_PARAMS, SEARCH_SEEDS)
    baseline_audit = repeated_group_oof(X, y, groups, DEFAULT_PARAMS, AUDIT_SEEDS)
    print(f"baseline search={baseline_search} audit={baseline_audit}")

    sampled_params = list(
        ParameterSampler(PARAM_DISTRIBUTIONS, n_iter=64, random_state=RANDOM_STATE)
    )
    sampled_params.insert(0, DEFAULT_PARAMS)

    search_results = []
    for idx, params in enumerate(sampled_params):
        fold_mean, fold_std, oof_accuracy = repeated_group_oof(X, y, groups, params, SEARCH_SEEDS)
        item = {
            "rank_input": idx,
            "params": params,
            "search_fold_mean": fold_mean,
            "search_fold_std": fold_std,
            "search_oof_accuracy": oof_accuracy,
        }
        search_results.append(item)
        print(
            f"search {idx:03d}: oof={oof_accuracy:.6f} fold_mean={fold_mean:.6f} "
            f"leaves={params['num_leaves']} lr={params['learning_rate']} n={params['n_estimators']}"
        )

    top_search = sorted(search_results, key=lambda item: item["search_oof_accuracy"], reverse=True)[:10]
    audited = []
    for item in top_search:
        params = item["params"]
        fold_mean, fold_std, oof_accuracy = repeated_group_oof(X, y, groups, params, AUDIT_SEEDS)
        audited_item = {
            **item,
            "audit_fold_mean": fold_mean,
            "audit_fold_std": fold_std,
            "audit_oof_accuracy": oof_accuracy,
            "audit_delta_vs_default": oof_accuracy - baseline_audit[2],
            "search_delta_vs_default": item["search_oof_accuracy"] - baseline_search[2],
        }
        audited.append(audited_item)
        print(
            f"audit input={item['rank_input']:03d}: audit_oof={oof_accuracy:.6f} "
            f"delta={audited_item['audit_delta_vs_default']:.6f}"
        )

    audited.sort(
        key=lambda item: (
            item["audit_oof_accuracy"],
            item["search_oof_accuracy"],
            -abs(item["search_oof_accuracy"] - item["audit_oof_accuracy"]),
        ),
        reverse=True,
    )
    winner = audited[0]
    submission = full_fit_submission(train_df, test_df, winner["params"], seed=42)
    submission.to_csv(SUBMISSION_PATH, index=False)

    payload = {
        "protocol": {
            "cv_scheme": "Repeated StratifiedGroupKFold by Passenger group",
            "feature_set": "simple 22-feature, no surname, no test-label leak",
            "search_seeds": SEARCH_SEEDS,
            "audit_seeds": AUDIT_SEEDS,
            "selection": "Random search on search seeds, final choice by independent audit seeds.",
            "final_training": "Full train fit with random_state=42, matching the known best submission style.",
        },
        "default_params": DEFAULT_PARAMS,
        "baseline": {
            "search_fold_mean": baseline_search[0],
            "search_fold_std": baseline_search[1],
            "search_oof_accuracy": baseline_search[2],
            "audit_fold_mean": baseline_audit[0],
            "audit_fold_std": baseline_audit[1],
            "audit_oof_accuracy": baseline_audit[2],
        },
        "winner": {
            **winner,
            "full_fit_seed": 42,
            "submission_path": str(SUBMISSION_PATH),
            "submission_true_rate": float(submission["Transported"].mean()),
            "compare_to_best_known": compare_to_best_known(submission),
        },
        "top_audited": audited,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["winner"], indent=2))


if __name__ == "__main__":
    main()
