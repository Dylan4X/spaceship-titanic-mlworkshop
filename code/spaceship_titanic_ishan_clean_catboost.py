from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from project_paths import DATA_DIR, EXPERIMENTS_TABLE_DIR, SUBMISSIONS_DIR

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = SUBMISSIONS_DIR / "submission_ishan_clean_catboost.csv"
RESULTS_PATH = EXPERIMENTS_TABLE_DIR / "ishan_clean_catboost_results.json"

SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
SHOPPING = ["FoodCourt", "ShoppingMall"]
SERVICE = ["RoomService", "Spa", "VRDeck"]
CATBOOST_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.01,
    "depth": 10,
    "l2_leaf_reg": 5,
    "border_count": 64,
    "loss_function": "Logloss",
    "random_seed": 42,
    "verbose": False,
}


def safe_mode(values: pd.Series, default=np.nan):
    mode = values.dropna().mode()
    if mode.empty:
        return default
    return mode.iloc[0]


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse=False)


def add_base_features(df_train: pd.DataFrame, df_test: pd.DataFrame):
    train = df_train.copy()
    test = df_test.copy()

    for df in [train, test]:
        df["Total_Spending"] = df[SPEND].sum(axis=1)
        df["No_spending"] = (df["Total_Spending"] == 0).astype(int)
        df["UsedAmenities"] = df[SPEND].gt(0).sum(axis=1)
        df["Service_Spending"] = df[SERVICE].sum(axis=1)
        df["Shopping_Spending"] = df[SHOPPING].sum(axis=1)
        df.drop(SPEND, axis=1, inplace=True)

        df["Age_group"] = np.nan
        df.loc[df["Age"] <= 12, "Age_group"] = "Age_0-12"
        df.loc[(df["Age"] > 12) & (df["Age"] < 18), "Age_group"] = "Age_13-17"
        df.loc[(df["Age"] >= 18) & (df["Age"] <= 25), "Age_group"] = "Age_18-25"
        df.loc[(df["Age"] > 25) & (df["Age"] <= 30), "Age_group"] = "Age_26-30"
        df.loc[(df["Age"] > 30) & (df["Age"] <= 50), "Age_group"] = "Age_31-50"
        df.loc[df["Age"] > 50, "Age_group"] = "Age_51+"

        df["Group"] = df["PassengerId"].astype(str).str[:4].astype(int)

    train_group_size = train["Group"].value_counts().to_dict()
    test_group_size = test["Group"].value_counts().to_dict()
    train["Group_Size"] = train["Group"].map(train_group_size)
    test["Group_Size"] = test["Group"].map(test_group_size)
    train["Solo"] = train["Group_Size"].eq(1)
    test["Solo"] = test["Group_Size"].eq(1)

    for df in [train, test]:
        df["Cabin"] = df["Cabin"].fillna("Z/9999/Z")
        cabin = df["Cabin"].str.split("/", expand=True)
        df["Cabin_deck"] = cabin[0]
        df["Cabin_number"] = cabin[1].astype(int)
        df["Cabin_side"] = cabin[2]
        df.loc[df["Cabin_deck"] == "Z", "Cabin_deck"] = np.nan
        df.loc[df["Cabin_number"] == 9999, "Cabin_number"] = np.nan
        df.loc[df["Cabin_side"] == "Z", "Cabin_side"] = np.nan
        df.drop("Cabin", axis=1, inplace=True)

        df["Cabin_region1"] = (df["Cabin_number"] < 300).astype(int)
        df["Cabin_region2"] = ((df["Cabin_number"] >= 300) & (df["Cabin_number"] < 600)).astype(int)
        df["Cabin_region3"] = ((df["Cabin_number"] >= 600) & (df["Cabin_number"] < 900)).astype(int)
        df["Cabin_region4"] = ((df["Cabin_number"] >= 900) & (df["Cabin_number"] < 1200)).astype(int)
        df["Cabin_region5"] = ((df["Cabin_number"] >= 1200) & (df["Cabin_number"] < 1500)).astype(int)
        df["Cabin_region6"] = ((df["Cabin_number"] >= 1500) & (df["Cabin_number"] < 1800)).astype(int)
        df["Cabin_region7"] = (df["Cabin_number"] >= 1800).astype(int)

        df["Name"] = df["Name"].fillna("Unknown Unknown")
        df["Surname"] = df["Name"].str.split().str[-1]

    all_surnames = pd.concat([train["Surname"], test["Surname"]])
    surname_counts = all_surnames.value_counts()
    train["Family_size"] = train["Surname"].map(surname_counts)
    test["Family_size"] = test["Surname"].map(surname_counts)

    for df in [train, test]:
        df.loc[df["Surname"] == "Unknown", "Surname"] = np.nan
        df.loc[df["Family_size"] > 100, "Family_size"] = np.nan
        df.drop("Name", axis=1, inplace=True)

    return train, test


def fill_group_mode(data: pd.DataFrame, feature: str, group_col: str = "Group") -> None:
    for idx, row in data.loc[data[feature].isna()].iterrows():
        value = safe_mode(data.loc[data[group_col] == row[group_col], feature])
        if pd.notna(value):
            data.at[idx, feature] = value


def preprocess() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series, ColumnTransformer]:
    df_train = pd.read_csv(DATA_DIR / "train.csv")
    df_test = pd.read_csv(DATA_DIR / "test.csv")
    passenger_ids = df_test["PassengerId"].copy()
    groups_for_cv = df_train["PassengerId"].astype(str).str[:4]

    df_train, df_test = add_base_features(df_train, df_test)
    y = df_train["Transported"].astype(int).copy()
    x_raw = df_train.drop("Transported", axis=1).copy()
    data = pd.concat([x_raw, df_test], axis=0).reset_index(drop=True)

    fill_group_mode(data, "HomePlanet")

    missing_home = data["HomePlanet"].isna()
    data.loc[missing_home & data["Cabin_deck"].isin(["A", "B", "C", "T"]), "HomePlanet"] = "Europa"
    data.loc[missing_home & data["Cabin_deck"].eq("G"), "HomePlanet"] = "Earth"

    for idx, row in data.loc[data["HomePlanet"].isna()].iterrows():
        value = safe_mode(data.loc[data["Surname"] == row["Surname"], "HomePlanet"])
        if pd.notna(value):
            data.at[idx, "HomePlanet"] = value

    data.loc[data["HomePlanet"].isna() & data["Cabin_deck"].eq("D"), "HomePlanet"] = "Mars"
    data.loc[data["HomePlanet"].isna(), "HomePlanet"] = "Earth"

    data["Destination"] = data["Destination"].fillna("TRAPPIST-1e")

    fill_group_mode(data, "Surname")
    data["Surname"] = data["Surname"].fillna("Unknown")
    family_counts = data["Surname"].value_counts()
    data["Family_size"] = data["Surname"].map(family_counts)
    data.loc[data["Surname"] == "Unknown", "Surname"] = np.nan
    data.loc[data["Family_size"] > 100, "Family_size"] = 0

    fill_group_mode(data, "Cabin_deck")
    fill_group_mode(data, "Cabin_side")

    for feature in ["Cabin_deck", "Cabin_side"]:
        na_idx = data.index[data[feature].isna()]
        grouped = data.groupby(["HomePlanet", "Destination", "Solo"])[feature].transform(
            lambda col: col.fillna(safe_mode(col))
        )
        data.loc[na_idx, feature] = grouped.loc[na_idx]

    for deck in ["A", "B", "C", "D", "E", "F", "G"]:
        known = data["Cabin_number"].notna() & data["Cabin_deck"].eq(deck)
        missing = data["Cabin_number"].isna() & data["Cabin_deck"].eq(deck)
        if known.sum() >= 2 and missing.any():
            model = LinearRegression()
            model.fit(data.loc[known, ["Group"]], data.loc[known, "Cabin_number"])
            data.loc[missing, "Cabin_number"] = model.predict(data.loc[missing, ["Group"]]).astype(int)

    data["Cabin_number"] = data["Cabin_number"].fillna(data["Cabin_number"].median())
    data["Cabin_deck"] = data["Cabin_deck"].fillna(safe_mode(data["Cabin_deck"], "F"))
    data["Cabin_side"] = data["Cabin_side"].fillna(safe_mode(data["Cabin_side"], "S"))

    data["VIP"] = data["VIP"].fillna(False)

    na_age = data.index[data["Age"].isna()]
    grouped_age = data.groupby(["HomePlanet", "No_spending", "Solo", "Cabin_deck"])["Age"].transform(
        lambda col: col.fillna(col.median())
    )
    data.loc[na_age, "Age"] = grouped_age.loc[na_age]
    data["Age"] = data["Age"].fillna(data["Age"].median())

    data["Age_group"] = data["Age_group"].fillna("Age_missing")

    for idx, row in data.loc[data["CryoSleep"].isna()].iterrows():
        value = safe_mode(data.loc[data["No_spending"] == row["No_spending"], "CryoSleep"])
        if pd.notna(value):
            data.at[idx, "CryoSleep"] = value
    data["CryoSleep"] = data["CryoSleep"].fillna(False)

    x = data.iloc[: len(df_train)].copy()
    x_test = data.iloc[len(df_train) :].copy()
    x.drop(["PassengerId", "Group", "Surname", "Cabin_number"], axis=1, inplace=True)
    x_test.drop(["PassengerId", "Group", "Surname", "Cabin_number"], axis=1, inplace=True)

    categorical_cols = [col for col in x.columns if x[col].dtype == "object"]
    transformer = ColumnTransformer(
        transformers=[("cat", make_one_hot_encoder(), categorical_cols)],
        remainder="passthrough",
    )
    x_enc = transformer.fit_transform(x)
    x_test_enc = transformer.transform(x_test)
    return x_enc, x_test_enc, y.to_numpy(), passenger_ids, groups_for_cv, transformer


def run_cv(name: str, splitter, x, y, groups=None):
    oof = np.zeros(len(y), dtype=float)
    scores = []
    losses = []
    split_iter = splitter.split(x, y, groups) if groups is not None else splitter.split(x, y)
    for fold, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(x[tr_idx], y[tr_idx])
        proba = model.predict_proba(x[va_idx])[:, 1]
        oof[va_idx] = proba
        pred = (proba >= 0.5).astype(int)
        scores.append(float(accuracy_score(y[va_idx], pred)))
        losses.append(float(log_loss(y[va_idx], proba)))
        print(f"{name} fold {fold}: acc={scores[-1]:.5f} logloss={losses[-1]:.5f}")
    return {
        "scores": scores,
        "losses": losses,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "logloss": float(np.mean(losses)),
        "oof_acc": float(accuracy_score(y, (oof >= 0.5).astype(int))),
    }, oof


def main() -> None:
    x, x_test, y, passenger_ids, groups_for_cv, _ = preprocess()
    print("encoded shapes", x.shape, x_test.shape)

    group_metrics, group_oof = run_cv("group", GroupKFold(n_splits=5), x, y, groups_for_cv)
    strat_metrics, strat_oof = run_cv(
        "stratified",
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        x,
        y,
    )

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(x, y)
    test_proba = model.predict_proba(x_test)[:, 1]
    submission = pd.DataFrame({"PassengerId": passenger_ids, "Transported": test_proba >= 0.5})
    submission.to_csv(OUT_PATH, index=False)
    np.save(ROOT / "ishan_clean_catboost_test_probs.npy", test_proba)
    np.save(ROOT / "ishan_clean_catboost_group_oof_probs.npy", group_oof)
    np.save(ROOT / "ishan_clean_catboost_strat_oof_probs.npy", strat_oof)

    base = pd.read_csv(SUBMISSIONS_DIR / "submission_catboost_pycaret_exact.csv")
    results = {
        "params": CATBOOST_PARAMS,
        "group_cv": group_metrics,
        "stratified_cv": strat_metrics,
        "true_count": int(submission["Transported"].sum()),
        "diff_vs_exact": int(submission["Transported"].ne(base["Transported"]).sum()),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {OUT_PATH}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
