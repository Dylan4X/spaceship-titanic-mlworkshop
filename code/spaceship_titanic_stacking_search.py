import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from project_paths import DATA_DIR, EXPERIMENTS_TABLE_DIR, SUBMISSIONS_DIR


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
REFERENCE_PATH = ROOT / "submission (3).csv"
RESULTS_PATH = EXPERIMENTS_TABLE_DIR / "stacking_search_results.json"
SELECTED_PATH = SUBMISSIONS_DIR / "submission_stacking_rebuild.csv"

N_SPLITS = 5
CV_SEEDS = [7, 21, 42, 87, 123]

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
    "FamilySize",
    "CabinRegion",
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
    "FamilySize",
]

CATEGORICAL_FEATURES = [col for col in FEATURE_COLUMNS if col not in NUMERIC_FEATURES]


def add_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [train_df.drop(columns=["Transported"]), test_df],
        axis=0,
        ignore_index=True,
    )
    group_split = combined["PassengerId"].str.split("_", expand=True)
    combined["GroupId"] = group_split[0]
    combined["GroupMember"] = pd.to_numeric(group_split[1], errors="coerce")
    combined["GroupSize"] = combined.groupby("GroupId")["PassengerId"].transform("count")
    combined["IsAlone"] = (combined["GroupSize"] == 1).astype(int)

    cabin_split = combined["Cabin"].fillna("Unknown/Unknown/Unknown").str.split("/", expand=True)
    combined["CabinDeck"] = cabin_split[0]
    combined["CabinNum"] = pd.to_numeric(cabin_split[1].replace("Unknown", np.nan), errors="coerce")
    combined["CabinSide"] = cabin_split[2]
    combined["CabinRegion"] = pd.cut(
        combined["CabinNum"],
        bins=[-1, 300, 600, 900, 1200, 1500, 1800, 10000],
        labels=["r1", "r2", "r3", "r4", "r5", "r6", "r7"],
    ).astype("object")

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in spend_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined["TotalSpend"] = combined[spend_cols].fillna(0).sum(axis=1)
    combined["NoSpend"] = (combined["TotalSpend"] == 0).astype(int)
    combined["LuxurySpend"] = combined[["Spa", "VRDeck", "FoodCourt"]].fillna(0).sum(axis=1)
    combined["BasicSpend"] = combined[["RoomService", "ShoppingMall"]].fillna(0).sum(axis=1)

    for col in ["CryoSleep", "VIP", "HomePlanet", "Destination"]:
        combined[col] = combined[col].astype("object")
    combined["AgeBand"] = pd.cut(
        combined["Age"],
        bins=[-1, 12, 18, 25, 40, 60, 200],
        labels=["child", "teen", "young_adult", "adult", "middle_age", "senior"],
    ).astype("object")
    combined["CryoNoSpendMatch"] = (
        combined["CryoSleep"].fillna("Unknown").eq("True") & (combined["NoSpend"] == 1)
    ).astype(int)

    surname = combined["Name"].fillna("Unknown").str.split().str[-1]
    family_size = surname.map(surname.value_counts())
    combined["FamilySize"] = family_size.mask(surname.eq("Unknown") | family_size.gt(100), np.nan)

    train_fe = combined.iloc[: len(train_df)].copy()
    train_fe["Transported"] = train_df["Transported"].values
    test_fe = combined.iloc[len(train_df) :].copy()
    return train_fe, test_fe


def ordinal_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
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


def onehot_preprocessor(with_scaling: bool = False) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if with_scaling:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def model_factories() -> dict:
    return {
        "lgb_default": lambda seed: Pipeline(
            [
                ("preprocessor", ordinal_preprocessor()),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=500,
                        learning_rate=0.03,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        min_child_samples=25,
                        reg_alpha=0.1,
                        reg_lambda=0.1,
                        random_state=seed,
                        objective="binary",
                        verbosity=-1,
                    ),
                ),
            ]
        ),
        "lgb_tuned": lambda seed: Pipeline(
            [
                ("preprocessor", ordinal_preprocessor()),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=1000,
                        learning_rate=0.03,
                        num_leaves=15,
                        max_depth=6,
                        subsample=1.0,
                        colsample_bytree=1.0,
                        min_child_samples=35,
                        reg_alpha=0.7,
                        reg_lambda=0.1,
                        random_state=seed,
                        objective="binary",
                        verbosity=-1,
                    ),
                ),
            ]
        ),
        "xgb_conservative": lambda seed: Pipeline(
            [
                ("preprocessor", ordinal_preprocessor()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=450,
                        learning_rate=0.035,
                        max_depth=3,
                        min_child_weight=4,
                        subsample=0.9,
                        colsample_bytree=0.85,
                        reg_alpha=0.2,
                        reg_lambda=2.0,
                        random_state=seed,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                    ),
                ),
            ]
        ),
        "extra_trees": lambda seed: Pipeline(
            [
                ("preprocessor", ordinal_preprocessor()),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=700,
                        max_depth=9,
                        min_samples_leaf=4,
                        max_features=0.8,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "random_forest": lambda seed: Pipeline(
            [
                ("preprocessor", ordinal_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=700,
                        max_depth=9,
                        min_samples_leaf=4,
                        max_features=0.8,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": lambda seed: Pipeline(
            [
                ("preprocessor", onehot_preprocessor()),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=250,
                        learning_rate=0.035,
                        max_depth=3,
                        min_samples_leaf=20,
                        subsample=0.9,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "logistic": lambda seed: Pipeline(
            [
                ("preprocessor", onehot_preprocessor(with_scaling=True)),
                (
                    "model",
                    LogisticRegression(
                        C=0.7,
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def predict_proba_or_score(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model[-1], "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-scores))


def build_oof_and_test(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, X_test: pd.DataFrame
) -> tuple[dict, dict]:
    oof_outputs = {}
    test_outputs = {}
    model_scores = {}

    for model_name, factory in model_factories().items():
        oof_sum = np.zeros(len(X), dtype=float)
        oof_counts = np.zeros(len(X), dtype=int)
        test_sum = np.zeros(len(X_test), dtype=float)
        fold_scores = []
        model_count = 0

        for seed in CV_SEEDS:
            splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            for train_idx, valid_idx in splitter.split(X, y, groups):
                model = factory(seed)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                valid_prob = predict_proba_or_score(model, X.iloc[valid_idx])
                test_prob = predict_proba_or_score(model, X_test)
                fold_scores.append(float(accuracy_score(y.iloc[valid_idx], valid_prob >= 0.5)))
                oof_sum[valid_idx] += valid_prob
                oof_counts[valid_idx] += 1
                test_sum += test_prob
                model_count += 1

        oof_outputs[model_name] = oof_sum / oof_counts
        test_outputs[model_name] = test_sum / model_count
        model_scores[model_name] = {
            "fold_mean": float(np.mean(fold_scores)),
            "fold_std": float(np.std(fold_scores)),
            "oof_accuracy": float(accuracy_score(y, oof_outputs[model_name] >= 0.5)),
        }
        print(model_name, model_scores[model_name])

    return {"oof": oof_outputs, "test": test_outputs}, model_scores


def top_k_prediction(prob: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(prob)
    pred = np.zeros(len(prob), dtype=bool)
    pred[order[-k:]] = True
    return pred


def main() -> None:
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df = add_features(raw_train, raw_test)
    X = train_df[FEATURE_COLUMNS]
    y = train_df["Transported"].astype(int)
    groups = train_df["GroupId"]
    X_test = test_df[FEATURE_COLUMNS]

    outputs, model_scores = build_oof_and_test(X, y, groups, X_test)
    oof_matrix = pd.DataFrame(outputs["oof"])
    test_matrix = pd.DataFrame(outputs["test"])

    stacker = LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")
    stacker.fit(oof_matrix, y)
    stack_oof = stacker.predict_proba(oof_matrix)[:, 1]
    stack_test = stacker.predict_proba(test_matrix)[:, 1]

    ridge = RidgeClassifier(alpha=10.0)
    ridge.fit(oof_matrix, y)
    ridge_oof = 1.0 / (1.0 + np.exp(-ridge.decision_function(oof_matrix)))
    ridge_test = 1.0 / (1.0 + np.exp(-ridge.decision_function(test_matrix)))

    ref = pd.read_csv(REFERENCE_PATH) if REFERENCE_PATH.exists() else None
    ref_true_count = int(ref["Transported"].sum()) if ref is not None else int(round(len(raw_test) * y.mean()))

    candidates = []
    candidate_probs = {
        "stack_logistic": stack_test,
        "stack_ridge": ridge_test,
        "mean_lgb_xgb_et": test_matrix[["lgb_default", "xgb_conservative", "extra_trees"]].mean(axis=1).values,
        "mean_all": test_matrix.mean(axis=1).values,
        "rank_lgb_tuned_default": 0.55 * test_matrix["lgb_default"].values + 0.45 * test_matrix["lgb_tuned"].values,
    }
    oof_probs = {
        "stack_logistic": stack_oof,
        "stack_ridge": ridge_oof,
        "mean_lgb_xgb_et": oof_matrix[["lgb_default", "xgb_conservative", "extra_trees"]].mean(axis=1).values,
        "mean_all": oof_matrix.mean(axis=1).values,
        "rank_lgb_tuned_default": 0.55 * oof_matrix["lgb_default"].values + 0.45 * oof_matrix["lgb_tuned"].values,
    }

    for name, prob in candidate_probs.items():
        for mode in ["threshold_05", "reference_count"]:
            pred = prob >= 0.5 if mode == "threshold_05" else top_k_prediction(prob, ref_true_count)
            oof_pred = oof_probs[name] >= 0.5
            if mode == "reference_count":
                oof_pred = top_k_prediction(oof_probs[name], int(y.sum()))
            diff_count = None
            if ref is not None:
                diff_count = int(pd.Series(pred).ne(ref["Transported"]).sum())
            candidates.append(
                {
                    "name": name,
                    "mode": mode,
                    "oof_accuracy": float(accuracy_score(y, oof_pred)),
                    "test_true_count": int(pred.sum()),
                    "test_true_rate": float(pred.mean()),
                    "diff_vs_reference": diff_count,
                }
            )

    candidates.sort(
        key=lambda item: (
            item["oof_accuracy"],
            -(item["diff_vs_reference"] if item["diff_vs_reference"] is not None else 0),
        ),
        reverse=True,
    )
    selected = next(
        item for item in candidates if item["name"] == "stack_logistic" and item["mode"] == "reference_count"
    )
    selected_prob = candidate_probs[selected["name"]]
    selected_pred = top_k_prediction(selected_prob, ref_true_count)
    submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Transported": selected_pred})
    submission.to_csv(SELECTED_PATH, index=False)

    payload = {
        "cv_scheme": "Repeated StratifiedGroupKFold, group by PassengerId prefix",
        "seeds": CV_SEEDS,
        "features": FEATURE_COLUMNS,
        "model_scores": model_scores,
        "stack_logistic_oof_accuracy": float(accuracy_score(y, stack_oof >= 0.5)),
        "stack_ridge_oof_accuracy": float(accuracy_score(y, ridge_oof >= 0.5)),
        "stacker_coefficients": dict(zip(oof_matrix.columns, stacker.coef_[0].tolist())),
        "reference_path": str(REFERENCE_PATH),
        "reference_true_count": ref_true_count,
        "candidates": candidates,
        "selected": {
            **selected,
            "path": str(SELECTED_PATH),
        },
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["selected"], indent=2))
    print(pd.DataFrame(candidates).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
