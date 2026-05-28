"""Reproduce the final public-best XGBoost submission.

This is the script version of the Kaggle demo notebook. On Kaggle it reads the
official competition files from `/kaggle/input` and writes the submission to
`/kaggle/working`. Locally it falls back to `data/` or the current directory.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import KMeansSMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle


SEED = 36086
PROJECT_ROOT = Path(__file__).resolve().parents[1]

XGB_PARAMS = {
    "lambda": 3.06,
    "alpha": 4.582,
    "colsample_bytree": 0.93,
    "subsample": 0.96,
    "n_estimators": 950,
    "max_depth": 5,
    "learning_rate": 0.0475,
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


def make_ohe() -> OneHotEncoder:
    """Support both older Kaggle sklearn and newer local sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def find_data_dir() -> Path:
    """Find official Kaggle CSVs in Kaggle or local repository layouts."""
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        for train_path in kaggle_root.rglob("train.csv"):
            data_dir = train_path.parent
            if (data_dir / "test.csv").exists() and (data_dir / "sample_submission.csv").exists():
                return data_dir

    candidates = [
        Path.cwd(),
        Path.cwd() / "data",
        PROJECT_ROOT,
        PROJECT_ROOT / "data",
    ]
    for data_dir in candidates:
        if all((data_dir / name).exists() for name in ("train.csv", "test.csv", "sample_submission.csv")):
            return data_dir

    raise FileNotFoundError("Could not find train.csv, test.csv, and sample_submission.csv.")


def prepare_frames(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the compact feature path used by the final public-best XGBoost branch."""
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    train_test = train._append(test, ignore_index=True)

    expense_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    train_test.loc[:, expense_cols] = train_test.apply(lambda row: 0 if row.CryoSleep == True else row, axis=1)
    train_test["Expenses"] = train_test.loc[:, expense_cols].sum(axis=1)
    train_test.loc[:, ["CryoSleep"]] = train_test.apply(
        lambda row: True if row.Expenses == 0 and pd.isna(row.CryoSleep) else row,
        axis=1,
    )

    train_test.Name = train_test.Name.fillna("Unknown Unknown")
    train_test.loc[:, ["Room"]] = train_test.PassengerId.apply(lambda value: value[0:4])

    guide_vip = train_test.loc[:, ["Room", "VIP"]].dropna().drop_duplicates("Room")
    guide_cabin = train_test.loc[:, ["Room", "Cabin"]].dropna().drop_duplicates("Room")
    guide_home = train_test.loc[:, ["Room", "HomePlanet"]].dropna().drop_duplicates("Room")
    guide_dest = train_test.loc[:, ["Room", "Destination"]].dropna().drop_duplicates("Room")

    train_test = pd.merge(train_test, guide_cabin, how="left", on="Room", suffixes=("", "_y"))
    train_test = pd.merge(train_test, guide_vip, how="left", on="Room", suffixes=("", "_y"))
    train_test = pd.merge(train_test, guide_home, how="left", on="Room", suffixes=("", "_y"))
    train_test = pd.merge(train_test, guide_dest, how="left", on="Room", suffixes=("", "_y"))

    train_test.loc[:, ["VIP"]] = train_test.apply(lambda row: row.VIP_y if pd.isna(row.VIP) else row, axis=1)
    train_test.loc[:, ["Cabin"]] = train_test.apply(lambda row: row.Cabin_y if pd.isna(row.Cabin) else row, axis=1)
    train_test.loc[:, ["HomePlanet"]] = train_test.apply(
        lambda row: row.HomePlanet_y if pd.isna(row.HomePlanet) else row,
        axis=1,
    )
    train_test.loc[:, ["Destination"]] = train_test.apply(
        lambda row: row.Destination_y if pd.isna(row.Destination) else row,
        axis=1,
    )

    train_test.loc[:, ["Cabin_1"]] = train_test.Cabin.str.split("/", expand=True).iloc[:, 0]
    train_test.loc[:, ["Cabin_2"]] = train_test.Cabin.str.split("/", expand=True).iloc[:, 1]
    train_test.loc[:, ["Cabin_3"]] = train_test.Cabin.str.split("/", expand=True).iloc[:, 2]
    train_test.loc[:, ["FirstName"]] = train_test.Name.str.split(" ", expand=True).iloc[:, 0]
    train_test.loc[:, ["SecondName"]] = train_test.Name.str.split(" ", expand=True).iloc[:, 1]
    train_test["Name_key"] = train_test["SecondName"] + train_test["Room"]

    num_cols = ["ShoppingMall", "FoodCourt", "RoomService", "Spa", "VRDeck", "Expenses", "Age"]
    cat_cols = ["CryoSleep", "Cabin_1", "Cabin_3", "VIP", "HomePlanet", "Destination"]
    train_test = train_test[num_cols + cat_cols + ["Transported"]].copy()

    train_test[num_cols] = pd.DataFrame(
        SimpleImputer(strategy="mean").fit_transform(train_test[num_cols]),
        columns=num_cols,
    )
    train_test[cat_cols] = pd.DataFrame(
        SimpleImputer(strategy="most_frequent").fit_transform(train_test[cat_cols]),
        columns=cat_cols,
    )
    ohe = make_ohe()
    encoded = pd.DataFrame(ohe.fit_transform(train_test[cat_cols]), columns=ohe.get_feature_names_out())
    train_test = pd.concat([train_test.drop(cat_cols, axis=1), encoded], axis=1)

    train_out = train_test[train_test["Transported"].notnull()].copy()
    train_out.Transported = train_out.Transported.astype("int")
    test_out = train_test[train_test["Transported"].isnull()].drop("Transported", axis=1)
    return train_out, test_out


def output_path() -> Path:
    kaggle_output = Path("/kaggle/working/Submission_XGB_0_81716.csv")
    if kaggle_output.parent.exists():
        return kaggle_output
    local_output = PROJECT_ROOT / "submissions" / "Submission_XGB_0_81716.csv"
    local_output.parent.mkdir(parents=True, exist_ok=True)
    return local_output


def main() -> None:
    data_dir = find_data_dir()
    print({"python": os.sys.version.split()[0], "xgboost": xgb.__version__, "data_dir": str(data_dir)})

    train_frame, test_frame = prepare_frames(data_dir)

    random.seed(SEED)
    np.random.seed(SEED)
    x = train_frame.drop("Transported", axis=1)
    y = train_frame.Transported
    x, y = shuffle(x, y)
    x = x.reset_index(drop=True).drop(DROP_LIST, axis=1)
    y = y.reset_index(drop=True)
    test_drop = test_frame.drop(DROP_LIST, axis=1)

    x_sm, y_sm = KMeansSMOTE(sampling_strategy=1, n_jobs=-1).fit_resample(x, y)
    print("Resampled class counts:", y_sm.value_counts().to_dict())

    pred = xgb.XGBClassifier(**XGB_PARAMS).fit(x_sm, y_sm).predict(test_drop)

    submission = pd.read_csv(data_dir / "sample_submission.csv")
    submission["Transported"] = pd.Series(pred).astype(bool)

    path = output_path()
    submission.to_csv(path, index=False)
    print("Saved:", path)
    print("Prediction counts:", submission["Transported"].value_counts().to_dict())


if __name__ == "__main__":
    main()
