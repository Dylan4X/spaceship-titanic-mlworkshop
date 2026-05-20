import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from project_paths import DATA_DIR, EXPERIMENTS_TABLE_DIR, SUBMISSIONS_DIR


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_catboost_pycaret_exact.csv"
RESULTS_PATH = EXPERIMENTS_TABLE_DIR / "pycaret_catboost_results.json"

SPEND_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


CATBOOST_PARAMS = {
    "nan_mode": "Min",
    "eval_metric": "Logloss",
    "iterations": 1000,
    "sampling_frequency": "PerTree",
    "leaf_estimation_method": "Newton",
    "grow_policy": "SymmetricTree",
    "penalties_coefficient": 1,
    "boosting_type": "Plain",
    "model_shrink_mode": "Constant",
    "feature_border_type": "GreedyLogSum",
    "l2_leaf_reg": 3,
    "random_strength": 1,
    "rsm": 1,
    "boost_from_average": False,
    "model_size_reg": 0.5,
    "subsample": 0.800000011920929,
    "use_best_model": False,
    "depth": 6,
    "posterior_sampling": False,
    "border_count": 254,
    "auto_class_weights": "None",
    "sparse_features_conflict_fraction": 0,
    "leaf_estimation_backtracking": "AnyImprovement",
    "best_model_min_trees": 1,
    "model_shrink_rate": 0,
    "min_data_in_leaf": 1,
    "loss_function": "Logloss",
    "learning_rate": 0.02582800015807152,
    "score_function": "Cosine",
    "task_type": "CPU",
    "leaf_estimation_iterations": 10,
    "bootstrap_type": "MVS",
    "max_leaves": 64,
    "random_seed": 7010,
    "verbose": False,
    "allow_writing_files": False,
}


def make_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    all_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    all_data["CryoSleep"] = all_data["CryoSleep"].fillna(False)
    all_data["Cabin"] = all_data["Cabin"].fillna("None")
    all_data["VIP"] = all_data["VIP"].fillna(all_data["VIP"].mode()[0])
    all_data["HomePlanet"] = all_data["HomePlanet"].fillna(all_data["HomePlanet"].mode()[0])
    all_data["Destination"] = all_data["Destination"].fillna(all_data["Destination"].mode()[0])
    all_data["Age"] = all_data["Age"].fillna(all_data["Age"].mean())
    all_data[SPEND_COLUMNS] = all_data[SPEND_COLUMNS].fillna(0)

    all_data["Deck"] = all_data["Cabin"].apply(lambda value: str(value)[:1])
    all_data["Side"] = all_data["Cabin"].apply(lambda value: str(value)[-1:])
    all_data["PassengerGroup"] = all_data["PassengerId"].apply(lambda value: value.split("_")[0])
    all_data["PassengerNo"] = all_data["PassengerId"].apply(lambda value: value.split("_")[1])

    all_data["TotalSpend"] = all_data[SPEND_COLUMNS].sum(axis=1)
    for col in SPEND_COLUMNS:
        all_data[f"Pct{col}"] = all_data[col] / all_data["TotalSpend"]
    all_data[[f"Pct{col}" for col in SPEND_COLUMNS]] = all_data[
        [f"Pct{col}" for col in SPEND_COLUMNS]
    ].fillna(0)

    all_data["AgeBin"] = 7
    for index in range(6):
        all_data.loc[
            (all_data["Age"] >= 10 * index) & (all_data["Age"] < 10 * (index + 1)),
            "AgeBin",
        ] = index

    all_data = all_data.drop(["PassengerId", "Name", "Cabin"], axis=1)
    for col in all_data.columns[all_data.dtypes == object]:
        if col != "Transported":
            encoder = LabelEncoder()
            all_data[col] = encoder.fit_transform(all_data[col])

    all_data["CryoSleep"] = all_data["CryoSleep"].astype(int)
    all_data["VIP"] = all_data["VIP"].astype(int)

    train_features = all_data.iloc[: len(train_df)].copy()
    test_features = all_data.iloc[len(train_df) :].drop(["Transported"], axis=1).copy()
    X = train_features.drop(["Transported"], axis=1)
    y = train_features["Transported"].astype(int)
    return X, y, test_features


def evaluate_cv(X: pd.DataFrame, y: pd.Series) -> dict:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=7010)
    scores = []
    for train_idx, valid_idx in splitter.split(X, y):
        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        valid_pred = model.predict(X.iloc[valid_idx]).astype(int)
        scores.append(float(accuracy_score(y.iloc[valid_idx], valid_pred)))
    return {
        "cv_scheme": "StratifiedKFold, matching the referenced PyCaret notebook",
        "fold_scores": scores,
        "mean_accuracy": float(np.mean(scores)),
        "std_accuracy": float(np.std(scores)),
    }


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    X, y, X_test = make_features(train_df, test_df)

    cv_metrics = evaluate_cv(X, y)
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(X, y)
    prediction = model.predict(X_test).astype(bool)

    submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Transported": prediction})
    submission.to_csv(SUBMISSION_PATH, index=False)

    payload = {
        "source": "Reimplementation of a Kaggle PyCaret/CatBoost notebook using local data.",
        "leakage_note": "Uses train+test feature columns only for unsupervised preprocessing; does not use test labels.",
        "features": X.columns.tolist(),
        "catboost_params": CATBOOST_PARAMS,
        "cv": cv_metrics,
        "submission_path": str(SUBMISSION_PATH),
        "submission_true_rate": float(submission["Transported"].mean()),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
