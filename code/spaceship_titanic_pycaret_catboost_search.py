import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold

from spaceship_titanic_pycaret_catboost import CATBOOST_PARAMS, make_features
from project_paths import DATA_DIR, EXPERIMENTS_TABLE_DIR, SUBMISSIONS_DIR


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
RESULTS_PATH = EXPERIMENTS_TABLE_DIR / "pycaret_catboost_search_results.json"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_catboost_pycaret_tuned.csv"

SEARCH_FOLDS = 3
AUDIT_FOLDS = 5
RANDOM_STATE = 20260411

PARAM_DISTRIBUTIONS = {
    "iterations": [800, 1000, 1200, 1400],
    "learning_rate": [0.018, 0.022, 0.02582800015807152, 0.03, 0.036, 0.045],
    "depth": [4, 5, 6, 7],
    "l2_leaf_reg": [2, 3, 4, 5, 7, 10],
    "subsample": [0.72, 0.8, 0.88, 0.95],
    "random_strength": [0.5, 1, 1.5, 2],
    "min_data_in_leaf": [1, 2, 4, 8],
    "rsm": [0.85, 0.95, 1.0],
    "bootstrap_type": ["MVS", "Bernoulli"],
}


def build_params(overrides: dict) -> dict:
    params = dict(CATBOOST_PARAMS)
    params.update(overrides)
    if overrides:
        # CatBoost only accepts max_leaves for Lossguide trees. The referenced
        # notebook includes it, but random-search variants can fail unless it is
        # removed for the symmetric-tree candidates we are exploring here.
        params.pop("max_leaves", None)
    params["random_seed"] = 7010
    params["verbose"] = False
    params["allow_writing_files"] = False
    return params


def evaluate(X: pd.DataFrame, y: pd.Series, params: dict, folds: int) -> dict:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=7010)
    scores = []
    for train_idx, valid_idx in splitter.split(X, y):
        model = CatBoostClassifier(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[valid_idx]).astype(int)
        scores.append(float(accuracy_score(y.iloc[valid_idx], pred)))
    return {
        "folds": folds,
        "mean_accuracy": float(np.mean(scores)),
        "std_accuracy": float(np.std(scores)),
        "fold_scores": scores,
    }


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    X, y, X_test = make_features(train_df, test_df)

    baseline_params = build_params({})
    baseline_search = evaluate(X, y, baseline_params, SEARCH_FOLDS)
    baseline_audit = evaluate(X, y, baseline_params, AUDIT_FOLDS)
    print("baseline", baseline_search, baseline_audit)

    sampled = list(ParameterSampler(PARAM_DISTRIBUTIONS, n_iter=28, random_state=RANDOM_STATE))
    sampled.insert(0, {})

    search_results = []
    for idx, overrides in enumerate(sampled):
        params = build_params(overrides)
        score = evaluate(X, y, params, SEARCH_FOLDS)
        item = {
            "idx": idx,
            "overrides": overrides,
            "search": score,
        }
        search_results.append(item)
        print(
            f"search {idx:03d}: mean={score['mean_accuracy']:.6f} std={score['std_accuracy']:.6f} "
            f"overrides={overrides}"
        )

    top = sorted(search_results, key=lambda item: item["search"]["mean_accuracy"], reverse=True)[:8]
    audited = []
    for item in top:
        params = build_params(item["overrides"])
        audit = evaluate(X, y, params, AUDIT_FOLDS)
        audited_item = {
            **item,
            "audit": audit,
            "audit_delta_vs_baseline": audit["mean_accuracy"] - baseline_audit["mean_accuracy"],
        }
        audited.append(audited_item)
        print(
            f"audit {item['idx']:03d}: mean={audit['mean_accuracy']:.6f} "
            f"delta={audited_item['audit_delta_vs_baseline']:.6f}"
        )

    audited.sort(
        key=lambda item: (
            item["audit"]["mean_accuracy"],
            item["search"]["mean_accuracy"],
            -item["audit"]["std_accuracy"],
        ),
        reverse=True,
    )
    winner = audited[0]
    winner_params = build_params(winner["overrides"])

    model = CatBoostClassifier(**winner_params)
    model.fit(X, y)
    prediction = model.predict(X_test).astype(bool)
    submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Transported": prediction})
    submission.to_csv(SUBMISSION_PATH, index=False)

    payload = {
        "baseline": {
            "params": baseline_params,
            "search": baseline_search,
            "audit": baseline_audit,
        },
        "winner": {
            **winner,
            "params": winner_params,
            "submission_path": str(SUBMISSION_PATH),
            "submission_true_rate": float(submission["Transported"].mean()),
        },
        "audited": audited,
        "search_results": search_results,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["winner"], indent=2))


if __name__ == "__main__":
    main()
