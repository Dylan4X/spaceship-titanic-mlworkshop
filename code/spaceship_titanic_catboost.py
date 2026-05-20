import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from project_paths import DATA_DIR, EXPERIMENTS_TABLE_DIR, SUBMISSIONS_DIR


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
REFERENCE_PATH = ROOT / "submission (3).csv"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_catboost.csv"
RESULTS_PATH = EXPERIMENTS_TABLE_DIR / "catboost_results.json"

N_SPLITS = 5
CV_SEEDS = [7, 21, 42, 87, 123]

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

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
    "CabinRegion",
    "FamilySize",
    "PctRoomService",
    "PctFoodCourt",
    "PctShoppingMall",
    "PctSpa",
    "PctVRDeck",
]

CATEGORICAL_FEATURES = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "CabinDeck",
    "CabinSide",
    "AgeBand",
    "CabinRegion",
]


def add_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([train_df.drop(columns=["Transported"]), test_df], ignore_index=True)

    group_split = combined["PassengerId"].str.split("_", expand=True)
    combined["GroupId"] = group_split[0]
    combined["GroupMember"] = pd.to_numeric(group_split[1], errors="coerce")
    combined["GroupSize"] = combined.groupby("GroupId")["PassengerId"].transform("count")
    combined["IsAlone"] = (combined["GroupSize"] == 1).astype(int)

    cabin_split = combined["Cabin"].fillna("Missing/Missing/Missing").str.split("/", expand=True)
    combined["CabinDeck"] = cabin_split[0]
    combined["CabinNum"] = pd.to_numeric(cabin_split[1].replace("Missing", np.nan), errors="coerce")
    combined["CabinSide"] = cabin_split[2]
    combined["CabinRegion"] = pd.cut(
        combined["CabinNum"],
        bins=[-1, 300, 600, 900, 1200, 1500, 1800, 10000],
        labels=["r1", "r2", "r3", "r4", "r5", "r6", "r7"],
    ).astype("object")

    for col in SPEND_COLS:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined["TotalSpend"] = combined[SPEND_COLS].fillna(0).sum(axis=1)
    combined["NoSpend"] = (combined["TotalSpend"] == 0).astype(int)
    combined["LuxurySpend"] = combined[["Spa", "VRDeck", "FoodCourt"]].fillna(0).sum(axis=1)
    combined["BasicSpend"] = combined[["RoomService", "ShoppingMall"]].fillna(0).sum(axis=1)

    denominator = combined["TotalSpend"].replace(0, np.nan)
    for col in SPEND_COLS:
        combined[f"Pct{col}"] = (combined[col] / denominator).fillna(0)

    combined["AgeBand"] = pd.cut(
        combined["Age"],
        bins=[-1, 12, 18, 25, 30, 50, 200],
        labels=["0_12", "13_17", "18_25", "26_30", "31_50", "51_plus"],
    ).astype("object")
    combined["CryoNoSpendMatch"] = (
        combined["CryoSleep"].fillna("Missing").astype(str).eq("True") & (combined["NoSpend"] == 1)
    ).astype(int)

    surname = combined["Name"].fillna("Missing").str.split().str[-1]
    family_size = surname.map(surname.value_counts())
    combined["FamilySize"] = family_size.mask(surname.eq("Missing") | family_size.gt(100), np.nan)

    for col in CATEGORICAL_FEATURES:
        combined[col] = combined[col].astype("object").where(combined[col].notna(), "Missing").astype(str)

    train_fe = combined.iloc[: len(train_df)].copy()
    train_fe["Transported"] = train_df["Transported"].values
    test_fe = combined.iloc[len(train_df) :].copy()
    return train_fe, test_fe


def make_model(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=900,
        learning_rate=0.035,
        depth=4,
        l2_leaf_reg=5.0,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def cat_indices() -> list[int]:
    return [FEATURE_COLUMNS.index(col) for col in CATEGORICAL_FEATURES]


def cv_predict(X: pd.DataFrame, y: pd.Series, groups: pd.Series, X_test: pd.DataFrame) -> tuple[dict, np.ndarray]:
    oof_prob_sum = np.zeros(len(X))
    oof_counts = np.zeros(len(X))
    test_prob_sum = np.zeros(len(X_test))
    fold_scores = []
    model_count = 0

    for seed in CV_SEEDS:
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups), start=1):
            model = make_model(seed)
            train_pool = Pool(X.iloc[train_idx], y.iloc[train_idx], cat_features=cat_indices())
            valid_pool = Pool(X.iloc[valid_idx], cat_features=cat_indices())
            test_pool = Pool(X_test, cat_features=cat_indices())
            model.fit(train_pool)
            valid_prob = model.predict_proba(valid_pool)[:, 1]
            test_prob = model.predict_proba(test_pool)[:, 1]
            score = float(accuracy_score(y.iloc[valid_idx], valid_prob >= 0.5))
            fold_scores.append(score)
            oof_prob_sum[valid_idx] += valid_prob
            oof_counts[valid_idx] += 1
            test_prob_sum += test_prob
            model_count += 1
            print(f"seed={seed} fold={fold} accuracy={score:.6f}")

    oof_prob = oof_prob_sum / oof_counts
    metrics = {
        "fold_mean": float(np.mean(fold_scores)),
        "fold_std": float(np.std(fold_scores)),
        "oof_accuracy": float(accuracy_score(y, oof_prob >= 0.5)),
    }
    return metrics, test_prob_sum / model_count


def main() -> None:
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df = add_features(raw_train, raw_test)
    X = train_df[FEATURE_COLUMNS]
    y = train_df["Transported"].astype(int)
    groups = train_df["GroupId"]
    X_test = test_df[FEATURE_COLUMNS]

    metrics, test_prob = cv_predict(X, y, groups, X_test)
    submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Transported": test_prob >= 0.5})
    submission.to_csv(SUBMISSION_PATH, index=False)

    reference_info = {}
    if REFERENCE_PATH.exists():
        reference = pd.read_csv(REFERENCE_PATH)
        reference_info = {
            "reference_true_rate": float(reference["Transported"].mean()),
            "diff_vs_reference": int(submission["Transported"].ne(reference["Transported"]).sum()),
        }

    payload = {
        "cv_scheme": "Repeated StratifiedGroupKFold, group by PassengerId prefix",
        "seeds": CV_SEEDS,
        "features": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_params": make_model(42).get_params(),
        "metrics": metrics,
        "submission_path": str(SUBMISSION_PATH),
        "submission_true_rate": float(submission["Transported"].mean()),
        **reference_info,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
