import json
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_DIR = PROJECT_ROOT / "paper_support" / "data"
NOTE_DIR = PROJECT_ROOT / "paper_support" / "notes"
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "tables"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

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
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def style_axis(ax, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, color="#D8DDE6", linewidth=0.5)
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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)


def add_clean_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grp = out["PassengerId"].astype(str).str.split("_", expand=True)
    out["PassengerGroup"] = grp[0]
    out["PassengerNo"] = pd.to_numeric(grp[1], errors="coerce")
    out["GroupSize"] = out.groupby("PassengerGroup")["PassengerId"].transform("count")
    out["IsAlone"] = (out["GroupSize"] == 1).astype(int)

    cabin = out["Cabin"].fillna("Missing/-1/Missing").str.split("/", expand=True)
    out["Deck"] = cabin[0]
    out["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    out["Side"] = cabin[2]

    for col in SPEND_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["TotalSpend"] = out[SPEND_COLS].fillna(0).sum(axis=1)
    out["NoSpend"] = (out["TotalSpend"] == 0).astype(int)
    out["LogTotalSpend"] = np.log1p(out["TotalSpend"])
    nonzero = out["TotalSpend"].replace(0, np.nan)
    for col in SPEND_COLS:
        suffix = col if col != "ShoppingMall" else "ShoppingMall"
        out[f"Pct{suffix}"] = out[col].fillna(0) / nonzero
    out[[f"Pct{col if col != 'ShoppingMall' else 'ShoppingMall'}" for col in SPEND_COLS]] = out[
        [f"Pct{col if col != 'ShoppingMall' else 'ShoppingMall'}" for col in SPEND_COLS]
    ].fillna(0)
    out["AgeBand"] = pd.cut(
        out["Age"],
        bins=[0, 12, 18, 25, 35, 50, 80],
        labels=["0-12", "13-18", "19-25", "26-35", "36-50", "51+"],
        include_lowest=True,
    ).astype("object")
    out["CryoNoSpendMatch"] = (
        ((out["CryoSleep"] == True) & (out["NoSpend"] == 1))
        | ((out["CryoSleep"] == False) & (out["NoSpend"] == 0))
    ).astype(int)
    return out


def prepare_catboost_frame(
    feat: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    base_cols = feature_cols + ["Transported"]
    frame = feat[base_cols].copy()
    groups = feat["PassengerGroup"].copy()
    y = frame["Transported"].astype(int).copy()
    X = frame.drop(columns=["Transported"]).copy()

    num_cols = [c for c in X.columns if c not in categorical_cols]
    num_cols = [c for c in num_cols if c != "PassengerGroup" or c in categorical_cols]

    for col in categorical_cols:
        X[col] = X[col].astype("object").fillna("Missing").astype(str)
    for col in X.columns:
        if col not in categorical_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(X[col].median())

    return X, y, groups, categorical_cols


def cross_validate_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    categorical_cols: list[str],
    params: dict,
    splitter,
    use_groups: bool,
) -> tuple[float, float]:
    cat_idx = [X.columns.get_loc(c) for c in categorical_cols]
    scores = []
    split_iter = splitter.split(X, y, groups=groups) if use_groups else splitter.split(X, y)
    for train_idx, valid_idx in split_iter:
        model = CatBoostClassifier(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], cat_features=cat_idx)
        pred = model.predict(X.iloc[valid_idx]).astype(int).reshape(-1)
        scores.append(accuracy_score(y.iloc[valid_idx], pred))
    return float(np.mean(scores)), float(np.std(scores))


def run_feature_ablation(train_df: pd.DataFrame) -> pd.DataFrame:
    existing = TABLE_DIR / "feature_ablation.csv"
    if existing.exists():
        return pd.read_csv(existing)

    feat = add_clean_features(train_df)
    with open(NOTE_DIR / "catboost_results.json", "r", encoding="utf-8") as f:
        catboost_note = json.load(f)
    params = {**catboost_note["model_params"], "thread_count": 4}

    base_cols = ["HomePlanet", "CryoSleep", "Destination", "Age", "VIP"] + SPEND_COLS
    base_cat = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
    cabin_cols = ["Deck", "CabinNum", "Side"]
    cabin_cat = ["Deck", "Side"]
    group_cols = ["PassengerGroup", "PassengerNo", "GroupSize", "IsAlone"]
    group_cat = ["PassengerGroup"]
    spend_cols = ["TotalSpend", "NoSpend", "LogTotalSpend", "PctRoomService", "PctFoodCourt", "PctShoppingMall", "PctSpa", "PctVRDeck"]
    rule_cols = ["AgeBand", "CryoNoSpendMatch"]
    rule_cat = ["AgeBand"]

    stages = [
        ("Raw official features", base_cols, base_cat),
        ("+ Cabin features", base_cols + cabin_cols, base_cat + cabin_cat),
        ("+ Group features", base_cols + cabin_cols + group_cols, base_cat + cabin_cat + group_cat),
        ("+ Spend summaries", base_cols + cabin_cols + group_cols + spend_cols, base_cat + cabin_cat + group_cat),
        ("+ Rule-consistency", base_cols + cabin_cols + group_cols + spend_cols + rule_cols, base_cat + cabin_cat + group_cat + rule_cat),
    ]

    rows = []
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for label, cols, cat_cols in stages:
        X, y, groups, cat_features = prepare_catboost_frame(feat, cols, cat_cols)
        mean_acc, std_acc = cross_validate_catboost(X, y, groups, cat_features, params, splitter, use_groups=True)
        rows.append(
            {
                "stage": label,
                "feature_count": len(cols),
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
            }
        )

    ablation = pd.DataFrame(rows)
    ablation.to_csv(TABLE_DIR / "feature_ablation.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.45, 2.85))
    x = np.arange(len(ablation))
    ax.plot(x, ablation["mean_accuracy"], marker="o", linewidth=1.8, color=COLORS["blue"])
    ax.fill_between(
        x,
        ablation["mean_accuracy"] - ablation["std_accuracy"],
        ablation["mean_accuracy"] + ablation["std_accuracy"],
        color=COLORS["blue"],
        alpha=0.12,
    )
    for idx, row in ablation.iterrows():
        ax.text(idx, row["mean_accuracy"] + 0.00055, f"{row['mean_accuracy']:.4f}", ha="center", fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Raw", "+Cabin", "+Group", "+Spend", "+Rule"], rotation=18)
    ax.set_ylabel("Mean group-CV accuracy")
    ax.set_title("Feature-family ablation (CatBoost)")
    style_axis(ax, grid_axis="y")
    save_figure(fig, FIG_DIR / "feature_ablation.png")
    return ablation


def run_validation_comparison(train_df: pd.DataFrame) -> pd.DataFrame:
    existing = TABLE_DIR / "validation_scheme_comparison.csv"
    if existing.exists():
        return pd.read_csv(existing)

    feat = add_clean_features(train_df)
    with open(NOTE_DIR / "catboost_results.json", "r", encoding="utf-8") as f:
        catboost_note = json.load(f)
    params = {**catboost_note["model_params"], "iterations": 500, "thread_count": 4}

    full_cols = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "Age",
        "VIP",
        *SPEND_COLS,
        "Deck",
        "CabinNum",
        "Side",
        "PassengerGroup",
        "PassengerNo",
        "GroupSize",
        "IsAlone",
        "TotalSpend",
        "NoSpend",
        "LogTotalSpend",
        "PctRoomService",
        "PctFoodCourt",
        "PctShoppingMall",
        "PctSpa",
        "PctVRDeck",
        "AgeBand",
        "CryoNoSpendMatch",
    ]
    cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "PassengerGroup", "AgeBand"]
    X, y, groups, cat_features = prepare_catboost_frame(feat, full_cols, cat_cols)

    row_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    group_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    row_mean, row_std = cross_validate_catboost(X, y, groups, cat_features, params, row_splitter, use_groups=False)
    group_mean, group_std = cross_validate_catboost(X, y, groups, cat_features, params, group_splitter, use_groups=True)

    result = pd.DataFrame(
        [
            {"scheme": "Row-level StratifiedKFold", "mean_accuracy": row_mean, "std_accuracy": row_std},
            {"scheme": "Group-aware StratifiedGroupKFold", "mean_accuracy": group_mean, "std_accuracy": group_std},
        ]
    )
    result["gap_vs_group"] = result["mean_accuracy"] - group_mean
    result.to_csv(TABLE_DIR / "validation_scheme_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    sns.barplot(data=result, x="scheme", y="mean_accuracy", palette=[COLORS["orange"], COLORS["green"]], ax=ax)
    for idx, row in result.iterrows():
        ax.text(idx, row["mean_accuracy"] + 0.00045, f"{row['mean_accuracy']:.4f}", ha="center", fontsize=6.8)
    ax.set_xlabel("")
    ax.set_ylabel("Mean CV accuracy")
    ax.set_title("Validation scheme matters")
    ax.tick_params(axis="x", rotation=12)
    style_axis(ax, grid_axis="y")
    save_figure(fig, FIG_DIR / "validation_scheme_comparison.png")
    return result


def prepare_compact_xgb_branch(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    train_test = pd.concat([train_df, test_df], ignore_index=True)
    mask = train_test["CryoSleep"] == True
    train_test.loc[mask, SPEND_COLS] = 0
    train_test["Expenses"] = train_test[SPEND_COLS].sum(axis=1)
    train_test.loc[(train_test["Expenses"] == 0) & (train_test["CryoSleep"].isna()), "CryoSleep"] = True
    train_test["Name"] = train_test["Name"].fillna("Unknown Unknown")
    train_test["Room"] = train_test["PassengerId"].astype(str).str.slice(0, 4)

    guide_map = {}
    for col in ["VIP", "Cabin", "HomePlanet", "Destination"]:
        guide = train_test[["Room", col]].dropna().drop_duplicates("Room")
        guide_map[col] = guide.set_index("Room")[col].to_dict()
        train_test[col] = train_test[col].fillna(train_test["Room"].map(guide_map[col]))

    cabin = train_test["Cabin"].fillna("Unknown/Unknown/Unknown").str.split("/", expand=True)
    train_test["Cabin_1"] = cabin[0]
    train_test["Cabin_2"] = cabin[1]
    train_test["Cabin_3"] = cabin[2]

    name = train_test["Name"].str.split(" ", expand=True)
    train_test["FirstName"] = name[0]
    train_test["SecondName"] = name[1]
    train_test["Name_key"] = train_test["SecondName"] + train_test["Room"]

    num_cols = ["ShoppingMall", "FoodCourt", "RoomService", "Spa", "VRDeck", "Expenses", "Age"]
    cat_cols = ["CryoSleep", "Cabin_1", "Cabin_3", "VIP", "HomePlanet", "Destination"]

    compact = train_test[num_cols + cat_cols + ["Transported"]].copy()
    compact[num_cols] = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(compact[num_cols]), columns=num_cols)
    compact[cat_cols] = pd.DataFrame(SimpleImputer(strategy="most_frequent").fit_transform(compact[cat_cols]), columns=cat_cols)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded = pd.DataFrame(ohe.fit_transform(compact[cat_cols]), columns=ohe.get_feature_names_out())
    compact = pd.concat([compact.drop(columns=cat_cols).reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)

    train_proc = compact[compact["Transported"].notnull()].copy()
    train_proc["Transported"] = train_proc["Transported"].astype(int)
    X = train_proc.drop(columns=["Transported"]).copy()
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
    X = X.drop(columns=drop_list).copy()
    y = train_proc["Transported"].copy()
    groups = train_df["PassengerId"].astype(str).str.split("_").str[0].reset_index(drop=True)
    return X, y, groups


def evaluate_xgb_params(X: pd.DataFrame, y: pd.Series, groups: pd.Series, name: str, params: dict) -> dict:
    row_scores = []
    row_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train_idx, valid_idx in row_splitter.split(X, y):
        model = XGBClassifier(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[valid_idx])
        row_scores.append(accuracy_score(y.iloc[valid_idx], pred))

    group_scores = []
    group_splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    for train_idx, valid_idx in group_splitter.split(X, y, groups=groups):
        model = XGBClassifier(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[valid_idx])
        group_scores.append(accuracy_score(y.iloc[valid_idx], pred))

    return {
        "candidate": name,
        "row_cv_mean": float(np.mean(row_scores)),
        "row_cv_std": float(np.std(row_scores)),
        "group_cv_mean": float(np.mean(group_scores)),
        "group_cv_std": float(np.std(group_scores)),
        "params": json.dumps(params, sort_keys=True),
    }


def run_xgb_neighborhood_search(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    existing = TABLE_DIR / "xgb_neighborhood_search.csv"
    if existing.exists():
        return pd.read_csv(existing)

    X, y, groups = prepare_compact_xgb_branch(train_df, test_df)

    base = {
        "reg_lambda": 3.06,
        "reg_alpha": 4.582,
        "colsample_bytree": 0.93,
        "subsample": 0.96,
        "n_estimators": 725,
        "max_depth": 5,
        "learning_rate": 0.05,
        "random_state": 42,
        "n_jobs": 4,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    candidates = [
        ("Final branch params", base),
        ("Depth 4", {**base, "max_depth": 4}),
        ("Depth 6", {**base, "max_depth": 6}),
        ("500 trees", {**base, "n_estimators": 500}),
        ("LR 0.03", {**base, "learning_rate": 0.03, "n_estimators": 900}),
        ("Heavier regularization", {**base, "reg_alpha": 6.0, "reg_lambda": 4.0}),
        ("Lean sampling", {**base, "subsample": 0.90, "colsample_bytree": 0.88}),
    ]

    rows = [evaluate_xgb_params(X, y, groups, name, params) for name, params in candidates]
    result = pd.DataFrame(rows).sort_values("row_cv_mean", ascending=False).reset_index(drop=True)
    result["row_delta_vs_final"] = result["row_cv_mean"] - result.loc[result["candidate"] == "Final branch params", "row_cv_mean"].iloc[0]
    result["group_delta_vs_final"] = result["group_cv_mean"] - result.loc[result["candidate"] == "Final branch params", "group_cv_mean"].iloc[0]
    result.to_csv(TABLE_DIR / "xgb_neighborhood_search.csv", index=False)

    display_df = result.copy().sort_values("row_cv_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(3.5, 3.25))
    ax.hlines(display_df["candidate"], display_df["group_cv_mean"], display_df["row_cv_mean"], color=COLORS["light_gray"], linewidth=2.0)
    ax.scatter(display_df["group_cv_mean"], display_df["candidate"], s=26, color=COLORS["orange"], label="Group-aware CV", zorder=3)
    ax.scatter(display_df["row_cv_mean"], display_df["candidate"], s=26, color=COLORS["blue"], label="Row-level CV", zorder=3)
    ax.set_xlabel("Accuracy")
    ax.set_title("XGBoost parameter neighborhood")
    style_axis(ax, grid_axis="x")
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, FIG_DIR / "xgb_neighborhood_search.png")
    return result


def run_branch_disagreement_analysis(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    overall_path = TABLE_DIR / "branch_disagreement_overview.csv"
    deck_path = TABLE_DIR / "branch_disagreement_deck.csv"
    behavior_path = TABLE_DIR / "branch_disagreement_behavior.csv"
    if overall_path.exists() and deck_path.exists() and behavior_path.exists():
        return {
            "overview": pd.read_csv(overall_path),
            "deck": pd.read_csv(deck_path),
            "behavior": pd.read_csv(behavior_path),
        }

    feat = add_clean_features(train_df)
    with open(NOTE_DIR / "catboost_results.json", "r", encoding="utf-8") as f:
        catboost_note = json.load(f)
    cat_params = {**catboost_note["model_params"], "thread_count": 4}

    cat_cols = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "Deck",
        "Side",
        "PassengerGroup",
        "AgeBand",
    ]
    full_cols = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "Age",
        "VIP",
        *SPEND_COLS,
        "Deck",
        "CabinNum",
        "Side",
        "PassengerGroup",
        "PassengerNo",
        "GroupSize",
        "IsAlone",
        "TotalSpend",
        "NoSpend",
        "LogTotalSpend",
        "PctRoomService",
        "PctFoodCourt",
        "PctShoppingMall",
        "PctSpa",
        "PctVRDeck",
        "AgeBand",
        "CryoNoSpendMatch",
    ]
    X_cat, y_cat, groups_cat, cat_features = prepare_catboost_frame(feat, full_cols, cat_cols)
    cat_idx = [X_cat.columns.get_loc(c) for c in cat_features]

    oof_cat = np.zeros(len(y_cat), dtype=float)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, valid_idx in sgkf.split(X_cat, y_cat, groups=groups_cat):
        model = CatBoostClassifier(**cat_params)
        model.fit(X_cat.iloc[train_idx], y_cat.iloc[train_idx], cat_features=cat_idx)
        oof_cat[valid_idx] = model.predict_proba(X_cat.iloc[valid_idx])[:, 1]

    X_xgb, y_xgb, groups_xgb = prepare_compact_xgb_branch(train_df, test_df)
    xgb_params = {
        "reg_lambda": 3.06,
        "reg_alpha": 4.582,
        "colsample_bytree": 0.93,
        "subsample": 0.96,
        "n_estimators": 725,
        "max_depth": 5,
        "learning_rate": 0.05,
        "random_state": 42,
        "n_jobs": 4,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    oof_xgb = np.zeros(len(y_xgb), dtype=float)
    for train_idx, valid_idx in sgkf.split(X_xgb, y_xgb, groups=groups_xgb):
        model = XGBClassifier(**xgb_params)
        model.fit(X_xgb.iloc[train_idx], y_xgb.iloc[train_idx])
        oof_xgb[valid_idx] = model.predict_proba(X_xgb.iloc[valid_idx])[:, 1]

    if len(oof_cat) != len(oof_xgb):
        raise ValueError("CatBoost and XGBoost disagreement views have inconsistent row counts.")

    analysis = feat[["Deck", "CryoSleep", "NoSpend", "GroupSize"]].copy()
    analysis["target"] = y_cat.to_numpy()
    analysis["pred_cat"] = oof_cat >= 0.5
    analysis["pred_xgb"] = oof_xgb >= 0.5
    analysis["disagree"] = analysis["pred_cat"] != analysis["pred_xgb"]
    analysis["cat_correct"] = analysis["pred_cat"] == analysis["target"]
    analysis["xgb_correct"] = analysis["pred_xgb"] == analysis["target"]
    analysis["Deck"] = analysis["Deck"].fillna("Missing")
    analysis["cryo_flag"] = analysis["CryoSleep"].map({True: "Cryo=True", False: "Cryo=False"}).fillna("Cryo=Missing")
    analysis["spend_flag"] = analysis["NoSpend"].map({1: "NoSpend", 0: "Spend>0"})
    analysis["behavior"] = analysis["cryo_flag"] + " | " + analysis["spend_flag"]

    overview = pd.DataFrame(
        [
            {
                "shared_group_audit_rows": int(len(analysis)),
                "overall_disagreement_rate": float(analysis["disagree"].mean()),
                "catboost_accuracy": float(analysis["cat_correct"].mean()),
                "xgboost_accuracy": float(analysis["xgb_correct"].mean()),
                "accuracy_gap_xgb_minus_cat": float(analysis["xgb_correct"].mean() - analysis["cat_correct"].mean()),
            }
        ]
    )

    deck_summary = (
        analysis.groupby("Deck", dropna=False, observed=False)
        .agg(
            n=("target", "size"),
            disagreement_rate=("disagree", "mean"),
            catboost_accuracy=("cat_correct", "mean"),
            xgboost_accuracy=("xgb_correct", "mean"),
        )
        .reset_index()
    )
    deck_summary["accuracy_gap_xgb_minus_cat"] = deck_summary["xgboost_accuracy"] - deck_summary["catboost_accuracy"]
    deck_summary = deck_summary.sort_values("disagreement_rate", ascending=False).reset_index(drop=True)

    behavior_order = [
        "Cryo=Missing | NoSpend",
        "Cryo=False | NoSpend",
        "Cryo=False | Spend>0",
        "Cryo=Missing | Spend>0",
        "Cryo=True | NoSpend",
    ]
    behavior_summary = (
        analysis.groupby("behavior", dropna=False, observed=False)
        .agg(
            n=("target", "size"),
            disagreement_rate=("disagree", "mean"),
            catboost_accuracy=("cat_correct", "mean"),
            xgboost_accuracy=("xgb_correct", "mean"),
        )
        .reset_index()
    )
    behavior_summary["accuracy_gap_xgb_minus_cat"] = (
        behavior_summary["xgboost_accuracy"] - behavior_summary["catboost_accuracy"]
    )
    behavior_summary["order"] = behavior_summary["behavior"].map({label: idx for idx, label in enumerate(behavior_order)})
    behavior_summary = behavior_summary.sort_values("order").drop(columns="order").reset_index(drop=True)

    overview.to_csv(overall_path, index=False)
    deck_summary.to_csv(deck_path, index=False)
    behavior_summary.to_csv(behavior_path, index=False)

    deck_plot = deck_summary[deck_summary["n"] >= 150].copy().sort_values("disagreement_rate", ascending=True)
    deck_plot["label"] = deck_plot["Deck"] + deck_plot["n"].map(lambda x: f" (n={x})")
    behavior_plot = behavior_summary.copy().sort_values("disagreement_rate", ascending=True)
    behavior_plot["label"] = behavior_plot["behavior"] + behavior_plot["n"].map(lambda x: f" (n={x})")

    fig, axes = plt.subplots(1, 2, figsize=(6.95, 2.9))
    for ax, plot_df, title in [
        (axes[0], deck_plot, "Disagreement by cabin deck"),
        (axes[1], behavior_plot, "Disagreement by cryosleep-spend regime"),
    ]:
        ax.barh(plot_df["label"], plot_df["disagreement_rate"], color=COLORS["blue"], alpha=0.9)
        for _, row in plot_df.iterrows():
            gap = row["accuracy_gap_xgb_minus_cat"]
            gap_color = COLORS["green"] if gap > 0 else COLORS["red"]
            ax.text(
                row["disagreement_rate"] + 0.004,
                row["label"],
                f"{row['disagreement_rate']:.1%} | " + r"$\Delta$" + f"acc {gap:+.1%}",
                va="center",
                fontsize=6.6,
                color=gap_color,
            )
        ax.set_xlabel("Prediction disagreement rate")
        ax.set_title(title)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        style_axis(ax, grid_axis="x")

    save_figure(fig, FIG_DIR / "branch_disagreement.png")
    return {"overview": overview, "deck": deck_summary, "behavior": behavior_summary}


def build_tuning_summary() -> pd.DataFrame:
    existing = TABLE_DIR / "tuning_summary.csv"
    if existing.exists():
        return pd.read_csv(existing)

    with open(NOTE_DIR / "simple_search_results.json", "r", encoding="utf-8") as f:
        lgb = json.load(f)
    with open(NOTE_DIR / "pycaret_catboost_search_results.json", "r", encoding="utf-8") as f:
        cat = json.load(f)

    summary = pd.DataFrame(
        [
            {
                "model_family": "LightGBM",
                "baseline_audit_accuracy": lgb["baseline"]["audit_oof_accuracy"],
                "winner_audit_accuracy": lgb["winner"]["audit_oof_accuracy"],
                "audit_delta": lgb["winner"]["audit_delta_vs_default"],
                "search_method": "Random search with independent audit seeds",
            },
            {
                "model_family": "CatBoost",
                "baseline_audit_accuracy": cat["baseline"]["audit"]["mean_accuracy"],
                "winner_audit_accuracy": cat["winner"]["audit"]["mean_accuracy"],
                "audit_delta": cat["winner"]["audit_delta_vs_baseline"],
                "search_method": "Local parameter sweep with audit folds",
            },
        ]
    )
    summary.to_csv(TABLE_DIR / "tuning_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.25, 2.35))
    sns.barplot(data=summary, x="model_family", y="audit_delta", palette=[COLORS["teal"], COLORS["green"]], ax=ax)
    for idx, row in summary.iterrows():
        ax.text(idx, row["audit_delta"] + 0.00006, f"+{row['audit_delta']:.4f}", ha="center", fontsize=6.8)
    ax.set_xlabel("")
    ax.set_ylabel("Audit gain vs baseline")
    ax.set_title("Clean-model tuning gains")
    style_axis(ax, grid_axis="y")
    save_figure(fig, FIG_DIR / "tuning_summary.png")
    return summary


def build_tuning_key_changes() -> pd.DataFrame:
    existing = TABLE_DIR / "tuning_key_changes.csv"
    if existing.exists():
        return pd.read_csv(existing)

    with open(NOTE_DIR / "simple_search_results.json", "r", encoding="utf-8") as f:
        lgb = json.load(f)
    with open(NOTE_DIR / "pycaret_catboost_search_results.json", "r", encoding="utf-8") as f:
        cat = json.load(f)

    rows = []
    lgb_keys = [
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "max_depth",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "min_child_samples",
    ]
    for key in lgb_keys:
        rows.append(
            {
                "model_family": "LightGBM",
                "parameter": key,
                "baseline_value": lgb["default_params"][key],
                "winner_value": lgb["winner"]["params"][key],
                "changed": lgb["default_params"][key] != lgb["winner"]["params"][key],
                "audit_gain": lgb["winner"]["audit_delta_vs_default"],
            }
        )

    cat_keys = [
        "iterations",
        "learning_rate",
        "depth",
        "l2_leaf_reg",
        "subsample",
        "rsm",
        "random_strength",
        "min_data_in_leaf",
    ]
    for key in cat_keys:
        rows.append(
            {
                "model_family": "CatBoost",
                "parameter": key,
                "baseline_value": cat["baseline"]["params"][key],
                "winner_value": cat["winner"]["params"][key],
                "changed": cat["baseline"]["params"][key] != cat["winner"]["params"][key],
                "audit_gain": cat["winner"]["audit_delta_vs_baseline"],
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "tuning_key_changes.csv", index=False)
    return table


def main() -> None:
    warnings.filterwarnings("ignore")
    set_plot_style()
    train_df, test_df = load_data()
    run_feature_ablation(train_df)
    run_validation_comparison(train_df)
    run_xgb_neighborhood_search(train_df, test_df)
    run_branch_disagreement_analysis(train_df, test_df)
    build_tuning_summary()
    build_tuning_key_changes()
    print("Supplementary experiments completed.")


if __name__ == "__main__":
    main()
