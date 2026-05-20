import json
import os
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import KMeansSMOTE
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import (
    ApiListSubmissionsRequest,
)
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from xgboost import XGBClassifier

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
SUPPORT_ROOT = PROJECT_ROOT / "paper_support"
DATA_DIR = SUPPORT_ROOT / "data"
SOURCE_TABLE_DIR = SUPPORT_ROOT / "tables"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "tables"
SUMMARY_JSON = ROOT / "paper_summary.json"

KAGGLE_CONFIG_DIR = PROJECT_ROOT
LEADERBOARD_DIR = PROJECT_ROOT / "leaderboard" / "unzipped"

TEAM_USERNAME = "dylanxue04"
TEAM_NAME = "EAP_Hater@MLW"
BEST_CLEAN_FILE = "submission_catboost_pycaret_exact.csv"
BEST_PUBLIC_FILE = "Submission_XGB.csv"

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERIC_CORR_COLS = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "TotalSpend",
    "NoSpend",
]

SINGLE_COL_W = 3.42
DOUBLE_COL_W = 7.05

COLORS = {
    "blue": "#4C78A8",
    "teal": "#72B7B2",
    "green": "#54A24B",
    "orange": "#F58518",
    "red": "#E45756",
    "purple": "#B279A2",
    "gray": "#9D9D9D",
    "light_gray": "#E6E8EB",
    "dark": "#2F3B4A",
}
MODEL_COLORS = {
    "CatBoost": "#54A24B",
    "XGBoost": "#4C78A8",
    "LightGBM": "#72B7B2",
    "Random Forest": "#F58518",
    "Logistic Regression": "#9D755D",
}


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.6,
            "axes.labelsize": 8.2,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.1,
            "legend.title_fontsize": 7.3,
            "grid.color": "#D8DDE6",
            "grid.linewidth": 0.5,
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def style_axis(ax, grid_axis: str = "y") -> None:
    if grid_axis in {"x", "y", "both"}:
        ax.grid(True, axis=grid_axis, color="#D8DDE6", linewidth=0.5)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#AEB6C2")
    ax.spines["bottom"].set_color("#AEB6C2")
    ax.tick_params(colors=COLORS["dark"])
    ax.xaxis.label.set_color(COLORS["dark"])
    ax.yaxis.label.set_color(COLORS["dark"])
    ax.title.set_color(COLORS["dark"])


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout(pad=0.45)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group_split = out["PassengerId"].astype(str).str.split("_", expand=True)
    out["PassengerGroup"] = group_split[0]
    out["PassengerNo"] = pd.to_numeric(group_split[1], errors="coerce")
    out["GroupSize"] = out.groupby("PassengerGroup")["PassengerId"].transform("count")
    out["IsAlone"] = (out["GroupSize"] == 1).astype(int)

    cabin_split = out["Cabin"].fillna("Unknown/Unknown/Unknown").str.split("/", expand=True)
    out["Deck"] = cabin_split[0]
    out["CabinNum"] = pd.to_numeric(cabin_split[1].replace("Unknown", np.nan), errors="coerce")
    out["Side"] = cabin_split[2]
    out["DeckSide"] = out["Deck"].astype(str) + "-" + out["Side"].astype(str)

    for col in SPEND_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["TotalSpend"] = out[SPEND_COLS].fillna(0).sum(axis=1)
    out["NoSpend"] = (out["TotalSpend"] == 0).astype(int)
    out["LogTotalSpend"] = np.log1p(out["TotalSpend"])
    out["TransportedLabel"] = out["Transported"].map({False: "No", True: "Yes"})
    return out


def build_eda_assets(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    feat = add_features(train_df)

    dataset_summary = pd.DataFrame(
        [
            ("Training rows", len(train_df)),
            ("Test rows", len(test_df)),
            ("Raw predictive columns", 13),
            ("Positive class rate", round(float(train_df["Transported"].mean()), 5)),
            ("Largest missing rate", round(float(train_df.isna().mean().max()), 5)),
            ("Median age", float(pd.to_numeric(train_df["Age"], errors="coerce").median())),
            ("Zero-spend rate", round(float((feat["TotalSpend"] == 0).mean()), 5)),
            ("Multi-passenger rate", round(float((feat["GroupSize"] > 1).mean()), 5)),
            (
                "Group-homogeneous rate",
                round(
                    float(
                        train_df.groupby(train_df["PassengerId"].str.split("_").str[0])["Transported"]
                        .nunique()
                        .eq(1)
                        .mean()
                    ),
                    5,
                ),
            ),
        ],
        columns=["statistic", "value"],
    )
    dataset_summary.to_csv(TABLE_DIR / "dataset_summary.csv", index=False)

    feature_groups = pd.DataFrame(
        [
            ("Original demographics", "HomePlanet, Destination, Age, VIP, CryoSleep"),
            ("Cabin-derived", "Deck, CabinNum, Side, DeckSide"),
            ("Passenger-group-derived", "PassengerGroup, PassengerNo, GroupSize, IsAlone"),
            ("Spend-derived", "TotalSpend, NoSpend, spend ratios, LogTotalSpend"),
            ("Rule-consistency", "CryoSleep-spend consistency and zero-spend indicators"),
        ],
        columns=["feature_family", "examples"],
    )
    feature_groups.to_csv(TABLE_DIR / "feature_groups.csv", index=False)

    missing_rate = (
        train_df.isna().mean().sort_values(ascending=False).rename("missing_rate").reset_index()
    )
    missing_rate.columns = ["feature", "missing_rate"]
    missing_rate["missing_pct"] = missing_rate["missing_rate"] * 100
    missing_rate.to_csv(TABLE_DIR / "missing_rate.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, 2.35))
    top_missing = missing_rate[missing_rate["missing_pct"] > 0].copy()
    y_pos = np.arange(len(top_missing))
    axes[0].hlines(y=y_pos, xmin=0, xmax=top_missing["missing_pct"], color=COLORS["light_gray"], linewidth=2.2)
    axes[0].scatter(top_missing["missing_pct"], y_pos, s=26, color=COLORS["blue"], zorder=3)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(top_missing["feature"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Missing ratio (%)")
    axes[0].set_title("Raw-feature completeness")
    style_axis(axes[0], grid_axis="x")
    axes[0].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))

    share_no = 1.0 - float(train_df["Transported"].mean())
    share_yes = float(train_df["Transported"].mean())
    axes[1].barh([0], [share_no], color=COLORS["red"], height=0.45, label="False")
    axes[1].barh([0], [share_yes], left=[share_no], color=COLORS["green"], height=0.45, label="True")
    axes[1].text(share_no / 2, 0, f"False\n{share_no:.1%}", ha="center", va="center", color="white", fontsize=7.5)
    axes[1].text(share_no + share_yes / 2, 0, f"True\n{share_yes:.1%}", ha="center", va="center", color="white", fontsize=7.5)
    axes[1].set_xlim(0, 1)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Share of training set")
    axes[1].set_title("Balanced target distribution")
    axes[1].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    style_axis(axes[1], grid_axis="x")
    save_figure(fig, FIG_DIR / "eda_overview.png")

    age_bins = pd.cut(
        feat["Age"],
        bins=[0, 12, 18, 25, 35, 50, 80],
        labels=["0-12", "13-18", "19-25", "26-35", "36-50", "51+"],
        include_lowest=True,
    )
    age_rate = (
        pd.DataFrame({"AgeBand": age_bins, "Transported": feat["Transported"]})
        .dropna()
        .groupby("AgeBand", observed=False)["Transported"]
        .mean()
        .reset_index()
    )

    home_rate = (
        feat.groupby("HomePlanet", dropna=False)["Transported"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    home_rate["HomePlanet"] = home_rate["HomePlanet"].fillna("Missing")

    deck_rate = (
        feat.groupby("Deck", dropna=False)["Transported"].mean().sort_values(ascending=False).reset_index()
    )
    deck_rate["Deck"] = deck_rate["Deck"].fillna("Missing")

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_W, 4.55))

    axes[0, 0].plot(
        age_rate["AgeBand"].astype(str),
        age_rate["Transported"],
        marker="o",
        linewidth=1.7,
        color=COLORS["blue"],
    )
    axes[0, 0].set_ylim(0.38, 0.70)
    axes[0, 0].set_ylabel("Transport rate")
    axes[0, 0].set_title("Age contributes nonlinear signal")
    style_axis(axes[0, 0], grid_axis="y")

    sns.ecdfplot(
        data=feat,
        x="LogTotalSpend",
        hue="TransportedLabel",
        palette={"No": COLORS["red"], "Yes": COLORS["green"]},
        linewidth=1.6,
        ax=axes[0, 1],
    )
    axes[0, 1].set_xlabel("log(1 + TotalSpend)")
    axes[0, 1].set_ylabel("Cumulative share")
    axes[0, 1].set_title("Spend separates the two classes")
    style_axis(axes[0, 1], grid_axis="both")
    axes[0, 1].legend(frameon=False, title="")

    sns.barplot(
        data=home_rate,
        x="HomePlanet",
        y="Transported",
        palette=[COLORS["teal"]] * len(home_rate),
        ax=axes[1, 0],
    )
    axes[1, 0].set_ylim(0, 0.75)
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("Transport rate")
    axes[1, 0].set_title("HomePlanet remains informative")
    axes[1, 0].tick_params(axis="x", rotation=10)
    style_axis(axes[1, 0], grid_axis="y")

    sns.barplot(
        data=deck_rate,
        x="Deck",
        y="Transported",
        palette=[COLORS["purple"]] * len(deck_rate),
        ax=axes[1, 1],
    )
    axes[1, 1].set_ylim(0, 0.80)
    axes[1, 1].set_xlabel("Deck")
    axes[1, 1].set_ylabel("Transport rate")
    axes[1, 1].set_title("Deck captures strong cabin structure")
    style_axis(axes[1, 1], grid_axis="y")
    save_figure(fig, FIG_DIR / "eda_core_patterns.png")

    group_hist = (
        feat["GroupSize"].clip(upper=7).value_counts(normalize=True).sort_index().rename("share").reset_index()
    )
    group_hist.columns = ["GroupSize", "share"]
    group_hist["GroupLabel"] = group_hist["GroupSize"].astype(int).astype(str)
    group_hist.loc[group_hist["GroupSize"] == 7, "GroupLabel"] = "7+"

    corr_df = feat[NUMERIC_CORR_COLS].fillna(0).corr()
    corr_df.to_csv(TABLE_DIR / "numeric_correlation.csv")

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, 2.55))
    sns.barplot(
        data=group_hist,
        x="GroupLabel",
        y="share",
        color=COLORS["blue"],
        ax=axes[0],
    )
    axes[0].set_xlabel("Passenger group size")
    axes[0].set_ylabel("Share of training rows")
    axes[0].set_title("Travel groups are common, but not dominant")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[0].text(
        0.02,
        0.94,
        "44.7% of rows belong to\nmulti-passenger groups",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=7.1,
        color=COLORS["dark"],
        bbox={"facecolor": "white", "edgecolor": "#D8DDE6", "boxstyle": "round,pad=0.2"},
    )
    style_axis(axes[0], grid_axis="y")

    sns.heatmap(
        corr_df,
        cmap=sns.diverging_palette(240, 15, as_cmap=True),
        center=0,
        square=True,
        cbar_kws={"shrink": 0.78},
        ax=axes[1],
    )
    axes[1].set_title("Spend aggregation is informative, not redundant")
    axes[1].tick_params(axis="x", rotation=40, labelsize=6.7)
    axes[1].tick_params(axis="y", labelsize=6.7)
    save_figure(fig, FIG_DIR / "eda_group_and_correlation.png")

    return {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "positive_rate": float(train_df["Transported"].mean()),
        "largest_missing_rate": float(train_df.isna().mean().max()),
        "multi_passenger_rate": float((feat["GroupSize"] > 1).mean()),
    }


def load_benchmark_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark = pd.read_csv(SOURCE_TABLE_DIR / "model_benchmark.csv")
    importance = pd.read_csv(SOURCE_TABLE_DIR / "catboost_feature_importance.csv")
    benchmark.to_csv(TABLE_DIR / "model_benchmark.csv", index=False)
    importance.to_csv(TABLE_DIR / "catboost_feature_importance.csv", index=False)
    return benchmark, importance


def build_benchmark_figures(benchmark: pd.DataFrame, importance: pd.DataFrame) -> None:
    benchmark = benchmark.copy()
    benchmark["cv_mean_accuracy"] = benchmark["cv_mean_accuracy"].astype(float)
    benchmark["cv_std_accuracy"] = benchmark["cv_std_accuracy"].astype(float)
    benchmark["fit_time_sec"] = benchmark["fit_time_sec"].astype(float)
    benchmark["artifact_size_mb"] = benchmark["artifact_size_mb"].astype(float)

    plot_df = benchmark.sort_values("cv_mean_accuracy", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.75))
    for idx, row in plot_df.iterrows():
        color = MODEL_COLORS[row["model"]]
        ax.hlines(idx, row["cv_mean_accuracy"] - row["cv_std_accuracy"] / 3, row["cv_mean_accuracy"] + row["cv_std_accuracy"] / 3, color=color, linewidth=2.0)
        ax.scatter(row["cv_mean_accuracy"], idx, s=32, color=color, zorder=3)
        ax.text(row["cv_mean_accuracy"] + 0.00045, idx, f"{row['cv_mean_accuracy']:.4f}", va="center", fontsize=7.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["model"])
    ax.set_xlabel("Mean CV accuracy")
    ax.set_title("Unified local benchmark")
    ax.set_xlim(0.796, 0.8152)
    style_axis(ax, grid_axis="x")
    save_figure(fig, FIG_DIR / "model_benchmark.png")

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.9))
    for _, row in benchmark.iterrows():
        size = 42 + min(row["artifact_size_mb"], 20) * 8
        color = MODEL_COLORS[row["model"]]
        ax.scatter(row["fit_time_sec"], row["cv_mean_accuracy"], s=size, color=color, alpha=0.88, edgecolor="white", linewidth=0.5)
        ax.annotate(
            f"{row['model']}\n{row['artifact_size_mb']:.1f} MB",
            (row["fit_time_sec"], row["cv_mean_accuracy"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=6.8,
            color=COLORS["dark"],
        )
    ax.set_xscale("log")
    ax.set_xlabel("Training time across CV (log scale, s)")
    ax.set_ylabel("Mean CV accuracy")
    ax.set_title("Accuracy-cost trade-off")
    style_axis(ax, grid_axis="both")
    save_figure(fig, FIG_DIR / "model_efficiency_tradeoff.png")

    top_imp = importance.head(10).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 3.0))
    ax.barh(top_imp["feature"], top_imp["importance"], color=COLORS["green"])
    ax.set_xlabel("CatBoost importance")
    ax.set_title("Top CatBoost feature importances")
    style_axis(ax, grid_axis="x")
    save_figure(fig, FIG_DIR / "catboost_feature_importance.png")


def fetch_all_submissions(competition: str) -> pd.DataFrame:
    os.environ["KAGGLE_CONFIG_DIR"] = str(KAGGLE_CONFIG_DIR)
    api = KaggleApi()
    api.authenticate()
    rows = []
    with api.build_kaggle_client() as kaggle:
        token = ""
        while True:
            request = ApiListSubmissionsRequest()
            request.competition_name = competition
            request.page_token = token
            request.page_size = 200
            response = kaggle.competitions.competition_api_client.list_submissions(request)
            submissions = response.submissions or []
            for sub in submissions:
                rows.append(
                    {
                        "file_name": sub.file_name,
                        "date": pd.to_datetime(sub.date),
                        "description": sub.description,
                        "public_score": float(sub.public_score) if sub.public_score else np.nan,
                        "private_score": float(sub.private_score) if sub.private_score else np.nan,
                        "status": str(sub.status),
                    }
                )
            token = response.next_page_token
            if not token:
                break
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    return df


def load_leaderboard() -> pd.DataFrame:
    csv_files = sorted(LEADERBOARD_DIR.glob("spaceship-titanic-publicleaderboard-*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No leaderboard CSV found in {LEADERBOARD_DIR}")
    df = pd.read_csv(csv_files[-1], encoding="utf-8-sig")
    df["Rank"] = df["Rank"].astype(int)
    df["Score"] = df["Score"].astype(float)
    return df


def evaluate_highscore_xgb_branch() -> dict:
    train, test = load_train_test()

    params = {
        "reg_lambda": 3.06,
        "reg_alpha": 4.582,
        "colsample_bytree": 0.93,
        "subsample": 0.96,
        "n_estimators": 725,
        "max_depth": 5,
        "learning_rate": 0.05,
        "random_state": 42,
        "n_jobs": 1,
        "eval_metric": "logloss",
        "verbosity": 0,
    }

    train_test = pd.concat([train, test], ignore_index=True)
    train_test.loc[:, SPEND_COLS] = train_test.apply(lambda x: 0 if x.CryoSleep is True else x, axis=1)
    train_test["Expenses"] = train_test[SPEND_COLS].sum(axis=1)
    train_test.loc[:, ["CryoSleep"]] = train_test.apply(
        lambda x: True if x.Expenses == 0 and pd.isna(x.CryoSleep) else x,
        axis=1,
    )
    train_test["Name"] = train_test["Name"].fillna("Unknown Unknown")
    train_test["Room"] = train_test["PassengerId"].astype(str).str.slice(0, 4)

    guide_cols = {
        "VIP": train_test[["Room", "VIP"]].dropna().drop_duplicates("Room"),
        "Cabin": train_test[["Room", "Cabin"]].dropna().drop_duplicates("Room"),
        "HomePlanet": train_test[["Room", "HomePlanet"]].dropna().drop_duplicates("Room"),
        "Destination": train_test[["Room", "Destination"]].dropna().drop_duplicates("Room"),
    }
    for key, guide in guide_cols.items():
        train_test = pd.merge(train_test, guide, how="left", on="Room", suffixes=("", "_guide"))
        train_test[key] = train_test[key].where(train_test[key].notna(), train_test[f"{key}_guide"])

    cabin_split = train_test["Cabin"].str.split("/", expand=True)
    train_test["Cabin_1"] = cabin_split[0]
    train_test["Cabin_2"] = cabin_split[1]
    train_test["Cabin_3"] = cabin_split[2]

    name_split = train_test["Name"].str.split(" ", expand=True)
    train_test["FirstName"] = name_split[0]
    train_test["SecondName"] = name_split[1]
    train_test["Name_key"] = train_test["SecondName"] + train_test["Room"]

    num_cols = ["ShoppingMall", "FoodCourt", "RoomService", "Spa", "VRDeck", "Expenses", "Age"]
    cat_cols = ["CryoSleep", "Cabin_1", "Cabin_3", "VIP", "HomePlanet", "Destination"]
    train_test = train_test[num_cols + cat_cols + ["Transported"]].copy()

    num_imp = SimpleImputer(strategy="mean")
    cat_imp = SimpleImputer(strategy="most_frequent")
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_test[num_cols] = pd.DataFrame(num_imp.fit_transform(train_test[num_cols]), columns=num_cols)
    train_test[cat_cols] = pd.DataFrame(cat_imp.fit_transform(train_test[cat_cols]), columns=cat_cols)
    encoded = pd.DataFrame(ohe.fit_transform(train_test[cat_cols]), columns=ohe.get_feature_names_out())
    train_test = pd.concat(
        [train_test.drop(columns=cat_cols).reset_index(drop=True), encoded.reset_index(drop=True)],
        axis=1,
    )

    train_proc = train_test[train_test["Transported"].notnull()].copy()
    train_proc["Transported"] = train_proc["Transported"].astype(int)
    X = train_proc.drop(columns=["Transported"]).copy()
    y = train_proc["Transported"].copy()
    X, y = shuffle(X, y, random_state=42)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    def cv_mean_std(x_df: pd.DataFrame, y_s: pd.Series, splitter) -> tuple[float, float]:
        scores = []
        for train_idx, valid_idx in splitter.split(x_df, y_s):
            model = XGBClassifier(**params)
            model.fit(x_df.iloc[train_idx], y_s.iloc[train_idx])
            pred = model.predict(x_df.iloc[valid_idx])
            scores.append(accuracy_score(y_s.iloc[valid_idx], pred))
        return float(np.mean(scores)), float(np.std(scores))

    row10_before, row10_before_std = cv_mean_std(X, y, StratifiedKFold(n_splits=10, shuffle=False))

    features_isolation = ["ShoppingMall", "FoodCourt", "RoomService", "Spa", "VRDeck", "Age"]
    isf = IsolationForest(n_jobs=-1, random_state=1, n_estimators=100, contamination=0.003)
    keep_mask = isf.fit_predict(X[features_isolation]) == 1
    X_filtered = X.loc[keep_mask].reset_index(drop=True)
    y_filtered = y.loc[keep_mask].reset_index(drop=True)
    row10_after_outlier, row10_after_outlier_std = cv_mean_std(
        X_filtered,
        y_filtered,
        StratifiedKFold(n_splits=10, shuffle=False),
    )

    drop_list = [
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
    X_pruned = X.drop(columns=drop_list).copy()
    row10_after_prune, row10_after_prune_std = cv_mean_std(
        X_pruned,
        y,
        StratifiedKFold(n_splits=10, shuffle=False),
    )

    original_groups = train["PassengerId"].astype(str).str.split("_").str[0].reset_index(drop=True)
    groups = original_groups.iloc[X_pruned.index].reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    group_scores = []
    for train_idx, valid_idx in sgkf.split(X_pruned, y, groups=groups):
        model = XGBClassifier(**params)
        model.fit(X_pruned.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X_pruned.iloc[valid_idx])
        group_scores.append(accuracy_score(y.iloc[valid_idx], pred))

    x_res, y_res = KMeansSMOTE(sampling_strategy=1, n_jobs=-1).fit_resample(X_pruned, y)

    stage_df = pd.DataFrame(
        [
            {
                "stage": "Compact preprocessing branch",
                "metric": "10-fold row-level reconstruction CV",
                "accuracy": row10_before,
                "std": row10_before_std,
            },
            {
                "stage": "+ IsolationForest filtering",
                "metric": "10-fold row-level reconstruction CV",
                "accuracy": row10_after_outlier,
                "std": row10_after_outlier_std,
            },
            {
                "stage": "+ Feature pruning",
                "metric": "10-fold row-level reconstruction CV",
                "accuracy": row10_after_prune,
                "std": row10_after_prune_std,
            },
            {
                "stage": "Strict group-aware audit",
                "metric": "5-fold StratifiedGroupKFold",
                "accuracy": float(np.mean(group_scores)),
                "std": float(np.std(group_scores)),
            },
        ]
    )
    stage_df.to_csv(TABLE_DIR / "xgb_branch_reconstruction.csv", index=False)

    params_payload = {
        "final_params": params,
        "feature_dimension_before_prune": int(X.shape[1]),
        "feature_dimension_after_prune": int(X_pruned.shape[1]),
        "rows_after_isolation_filter": int(len(X_filtered)),
        "rows_after_kmeans_smote": int(len(x_res)),
    }
    (TABLE_DIR / "xgb_branch_params.json").write_text(json.dumps(params_payload, indent=2), encoding="utf-8")

    return {
        "stage_table": stage_df,
        "params": params,
        "feature_dim_before_prune": int(X.shape[1]),
        "feature_dim_after_prune": int(X_pruned.shape[1]),
        "rows_after_filter": int(len(X_filtered)),
        "rows_after_smote": int(len(x_res)),
        "strict_group_accuracy": float(np.mean(group_scores)),
        "strict_group_std": float(np.std(group_scores)),
    }


def build_xgb_branch_figure(stage_df: pd.DataFrame, submissions: pd.DataFrame) -> pd.DataFrame:
    xgb_path = pd.DataFrame(
        [
            ("Single XGB v1", "submission_xgboost.csv", "2026-04-01", 0.79378),
            ("Single XGB revised", "submission_xgboost.csv", "2026-04-10", 0.80383),
            ("Tuned XGB (rejected)", "submission_xgboost_tuned.csv", "2026-04-10", 0.79752),
            ("XGB-assisted stack", "submission_probe_stack90_xgb10.csv", "2026-04-16", 0.81084),
            ("Final public-best XGB branch", "Submission_XGB.csv", "2026-04-26", 0.81716),
        ],
        columns=["label", "file_name", "date", "public_score"],
    )
    xgb_path["date"] = pd.to_datetime(xgb_path["date"])
    xgb_path.to_csv(TABLE_DIR / "xgb_public_path.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, 2.7))

    row_df = stage_df[stage_df["metric"].str.contains("row-level")].reset_index(drop=True)
    axes[0].plot(
        np.arange(len(row_df)),
        row_df["accuracy"],
        marker="o",
        linewidth=1.8,
        color=COLORS["blue"],
    )
    axes[0].fill_between(
        np.arange(len(row_df)),
        row_df["accuracy"] - row_df["std"],
        row_df["accuracy"] + row_df["std"],
        color=COLORS["blue"],
        alpha=0.12,
    )
    for idx, row in row_df.iterrows():
        axes[0].text(idx, row["accuracy"] + 0.00055, f"{row['accuracy']:.4f}", ha="center", fontsize=6.9)
    strict_row = stage_df[stage_df["stage"] == "Strict group-aware audit"].iloc[0]
    axes[0].scatter([len(row_df) - 1], [strict_row["accuracy"]], marker="D", s=34, color=COLORS["orange"], zorder=4)
    axes[0].text(
        len(row_df) - 1 + 0.04,
        strict_row["accuracy"] - 0.00115,
        f"strict group audit\n{strict_row['accuracy']:.4f}",
        fontsize=6.8,
        color=COLORS["orange"],
        ha="left",
    )
    axes[0].set_xticks(np.arange(len(row_df)))
    axes[0].set_xticklabels(["Compact", "+IF", "+Prune"])
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.8025, 0.8092)
    axes[0].set_title("Stage-wise reconstruction of the final branch")
    style_axis(axes[0], grid_axis="y")

    axes[1].plot(
        xgb_path["date"],
        xgb_path["public_score"],
        marker="o",
        linewidth=1.8,
        color=COLORS["purple"],
    )
    axes[1].scatter(
        xgb_path.iloc[-1]["date"],
        xgb_path.iloc[-1]["public_score"],
        s=42,
        color=COLORS["red"],
        zorder=4,
    )
    for _, row in xgb_path.iterrows():
        offset = (4, 4)
        if row["label"] == "Final public-best XGB branch":
            offset = (5, 8)
        axes[1].annotate(
            f"{row['label']}\n{row['public_score']:.5f}",
            (row["date"], row["public_score"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.6,
            color=COLORS["dark"],
        )
    axes[1].set_ylabel("Public leaderboard score")
    axes[1].set_title("Our XGBoost exploration milestones")
    axes[1].tick_params(axis="x", rotation=20)
    style_axis(axes[1], grid_axis="y")
    save_figure(fig, FIG_DIR / "xgb_branch_evolution.png")

    return xgb_path


def build_competition_assets(xgb_branch_summary: dict) -> dict:
    submissions = fetch_all_submissions("spaceship-titanic")
    submissions.to_csv(TABLE_DIR / "submission_history.csv", index=False)

    leaderboard = load_leaderboard()
    leaderboard.to_csv(TABLE_DIR / "public_leaderboard_snapshot.csv", index=False)

    team_row = leaderboard[leaderboard["TeamMemberUserNames"].astype(str).str.contains(TEAM_USERNAME, na=False)].iloc[0]
    exact_row = leaderboard[leaderboard["Score"].eq(0.81061)].sort_values("Rank").iloc[0]

    score_map = (
        submissions.dropna(subset=["public_score"])
        .groupby("file_name", as_index=False)["public_score"]
        .max()
        .set_index("file_name")["public_score"]
        .to_dict()
    )

    xgb_path = build_xgb_branch_figure(xgb_branch_summary["stage_table"], submissions)

    milestone_df = pd.DataFrame(
        [
            ("Early feature-engineered baseline", "submission (3).csv", score_map.get("submission (3).csv")),
            ("Best clean CatBoost exact", BEST_CLEAN_FILE, score_map.get(BEST_CLEAN_FILE)),
            ("Flaykaer CatBoost reproduction", "submission_flaykaer_catboost_exact_repro.csv", score_map.get("submission_flaykaer_catboost_exact_repro.csv")),
            ("High-CV CatBoost counterexample", "submission_opamusora_clean_missing_foldavg_th050.csv", score_map.get("submission_opamusora_clean_missing_foldavg_th050.csv")),
            ("XGB-assisted stack", "submission_probe_stack90_xgb10.csv", score_map.get("submission_probe_stack90_xgb10.csv")),
            ("Final public-best XGB branch", BEST_PUBLIC_FILE, score_map.get(BEST_PUBLIC_FILE)),
        ],
        columns=["milestone", "file_name", "public_score"],
    )
    milestone_df.to_csv(TABLE_DIR / "milestones.csv", index=False)

    cv_public = pd.DataFrame(
        [
            ("LightGBM baseline", 0.81226, score_map.get("submission (3).csv")),
            ("CatBoost exact", 0.81330, score_map.get(BEST_CLEAN_FILE)),
            ("Flaykaer reproduction", 0.81422, score_map.get("submission_flaykaer_catboost_exact_repro.csv")),
            ("Opamusora clean", 0.81905, score_map.get("submission_opamusora_clean_missing_foldavg_th050.csv")),
            ("Public-best XGB branch", xgb_branch_summary["strict_group_accuracy"], score_map.get(BEST_PUBLIC_FILE)),
        ],
        columns=["label", "local_cv", "public_score"],
    )
    cv_public.to_csv(TABLE_DIR / "cv_vs_public.csv", index=False)

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.85))
    label_colors = {
        "LightGBM baseline": COLORS["teal"],
        "CatBoost exact": COLORS["green"],
        "Flaykaer reproduction": COLORS["blue"],
        "Opamusora clean": COLORS["gray"],
        "Public-best XGB branch": COLORS["red"],
    }
    for _, row in cv_public.iterrows():
        ax.scatter(row["local_cv"], row["public_score"], s=36, color=label_colors[row["label"]], zorder=3)
        ax.annotate(
            row["label"],
            (row["local_cv"], row["public_score"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=6.8,
            color=COLORS["dark"],
        )
    ax.set_xlabel("Local validation accuracy")
    ax.set_ylabel("Public leaderboard score")
    ax.set_title("Local validation and public score diverge")
    style_axis(ax, grid_axis="both")
    save_figure(fig, FIG_DIR / "cv_vs_public.png")

    plot_df = submissions.dropna(subset=["public_score"]).copy()
    plot_df = plot_df[plot_df["public_score"] >= 0.79].sort_values("date")
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 2.4))
    ax.plot(plot_df["date"], plot_df["public_score"], color=COLORS["blue"], linewidth=1.45, alpha=0.92)
    ax.scatter(plot_df["date"], plot_df["public_score"], s=13, color=COLORS["blue"])
    best_row = plot_df.loc[plot_df["public_score"].idxmax()]
    ax.scatter(best_row["date"], best_row["public_score"], s=46, color=COLORS["red"], zorder=4)
    ax.annotate(
        f"Best public score = {best_row['public_score']:.5f}",
        (best_row["date"], best_row["public_score"]),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=6.9,
    )
    ax.set_xlabel("Submission date")
    ax.set_ylabel("Public score")
    ax.set_title("Public-leaderboard progress over time")
    ax.tick_params(axis="x", rotation=20)
    style_axis(ax, grid_axis="y")
    save_figure(fig, FIG_DIR / "submission_timeline.png")

    score_dist = leaderboard["Score"].copy()
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.6))
    sns.histplot(score_dist, bins=40, color=COLORS["light_gray"], edgecolor="white", ax=ax)
    ax.axvline(float(exact_row["Score"]), color=COLORS["green"], linewidth=1.8)
    ax.axvline(float(team_row["Score"]), color=COLORS["red"], linewidth=1.8)
    ax.text(
        float(exact_row["Score"]) + 0.00003,
        ax.get_ylim()[1] * 0.82,
        f"Best clean single model\n0.81061 (Rank {int(exact_row['Rank'])})",
        color=COLORS["green"],
        fontsize=6.8,
    )
    ax.text(
        float(team_row["Score"]) + 0.00003,
        ax.get_ylim()[1] * 0.60,
        f"Our team\n0.81716 (Rank {int(team_row['Rank'])})",
        color=COLORS["red"],
        fontsize=6.8,
    )
    ax.set_xlabel("Public leaderboard score")
    ax.set_ylabel("Number of teams")
    ax.set_title("Leaderboard-score distribution")
    style_axis(ax, grid_axis="y")
    save_figure(fig, FIG_DIR / "leaderboard_context.png")

    selected_snapshot = pd.DataFrame(
        [
            ("Clean CatBoost exact line", "2026-04-11 09:36", 0.81061, "Clean anchor"),
            ("Flaykaer clean reproduction", "2026-04-12 16:09", 0.80991, "Clean reproduction"),
            ("High-CV CatBoost counterexample", "2026-04-12 17:34", 0.80593, "Validation counterexample"),
            ("XGB-assisted stack", "2026-04-16 11:57", 0.81084, "Competition-side hybrid"),
            ("Final public-best XGB branch", "2026-04-26 08:10", 0.81716, "Public-best branch"),
            ("Late blend attempt", "2026-05-14 09:37", 0.80967, "Late check"),
        ],
        columns=["milestone", "date", "public_score", "role"],
    )
    selected_snapshot["public_score"] = selected_snapshot["public_score"].map(lambda x: f"{x:.5f}")
    selected_snapshot.to_csv(TABLE_DIR / "selected_submission_snapshot.csv", index=False)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 2.35))
    ax.axis("off")
    table = ax.table(
        cellText=selected_snapshot[["milestone", "date", "public_score", "role"]].values,
        colLabels=["Milestone", "Submission time", "Public score", "Role"],
        colLoc="center",
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1, 1.14)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor(COLORS["dark"])
        else:
            cell.set_facecolor("#F7F9FB" if row % 2 else "#EDF2F7")
            cell.set_edgecolor("white")
    ax.set_title("Representative Kaggle submission milestones retrieved on 2026-05-17", fontsize=8.2, pad=7)
    save_figure(fig, FIG_DIR / "submission_log_snapshot.png")

    final_rank = pd.DataFrame(
        [
            ("Team name", str(team_row["TeamName"])),
            ("Team rank", int(team_row["Rank"])),
            ("Public score", float(team_row["Score"])),
            ("Leaderboard snapshot date", "2026-05-17"),
            ("Submission count", int(team_row["SubmissionCount"])),
            ("Total ranked teams", int(len(leaderboard))),
            ("Best clean single-model rank", int(exact_row["Rank"])),
            ("Best clean single-model public score", float(exact_row["Score"])),
        ],
        columns=["field", "value"],
    )
    final_rank.to_csv(TABLE_DIR / "final_ranking.csv", index=False)

    return {
        "team_name": str(team_row["TeamName"]),
        "team_rank": int(team_row["Rank"]),
        "team_score": float(team_row["Score"]),
        "team_members": str(team_row["TeamMemberUserNames"]),
        "total_teams": int(len(leaderboard)),
        "clean_single_rank": int(exact_row["Rank"]),
        "clean_single_score": float(exact_row["Score"]),
        "xgb_public_path": [
            {
                **row,
                "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            }
            for row in xgb_path.to_dict(orient="records")
        ],
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_dirs()
    set_plot_style()

    train_df, test_df = load_train_test()
    eda_summary = build_eda_assets(train_df, test_df)
    benchmark, importance = load_benchmark_tables()
    build_benchmark_figures(benchmark, importance)
    xgb_branch_summary = evaluate_highscore_xgb_branch()
    competition_summary = build_competition_assets(xgb_branch_summary)

    summary = {
        "eda": eda_summary,
        "xgb_branch": {
            "strict_group_accuracy": xgb_branch_summary["strict_group_accuracy"],
            "strict_group_std": xgb_branch_summary["strict_group_std"],
            "feature_dim_before_prune": xgb_branch_summary["feature_dim_before_prune"],
            "feature_dim_after_prune": xgb_branch_summary["feature_dim_after_prune"],
            "rows_after_filter": xgb_branch_summary["rows_after_filter"],
            "rows_after_smote": xgb_branch_summary["rows_after_smote"],
            "params": xgb_branch_summary["params"],
        },
        "competition": competition_summary,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
