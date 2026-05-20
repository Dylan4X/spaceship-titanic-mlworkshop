# Experiment Audit Against the Final Paper

## Verdict

The current experiment evidence is sufficient for the final paper and course rubric. I do not recommend adding a large new tuning campaign before submission because the paper's central claim is not "Optuna found a uniquely optimal model"; it is "validation-first evidence and public leaderboard behavior diverged, so the clean CatBoost anchor and public-best XGBoost branch should be reported separately."

## Tuning Evidence Already Present

- `tuning_summary.csv`: clean LightGBM and CatBoost tuning gains.
- `tuning_key_changes.csv`: concrete before/after hyperparameter changes for LightGBM and CatBoost.
- `xgb_neighborhood_search.csv`: focused neighborhood search around the public-best XGBoost parameters.
- `xgb_branch_reconstruction.csv`: stage-wise reconstruction of compact preprocessing, outlier diagnostic, feature pruning, and strict group-aware audit.
- `pipeline_audit.csv`: public-vs-local comparison for representative candidate lines.

## What Was Added for Demo Readiness

- `code/reproduce_highscore_xgb.py`: command-line reproduction of the high-score XGBoost branch.
- `notebooks/demo_repro_optuna_xgb_space_titanic.ipynb`: cleaned demo notebook.
- The demo path now assumes `optuna` is installed and runs a small Optuna search by default.

## Remaining Non-Code Placeholders

- `report/final_ieee_paper.tex` still contains a placeholder for the final GitHub URL.
- `report/final_ieee_paper.tex` still contains a placeholder for official student names/IDs and member-level contribution mapping.

Those are administrative facts, not modeling gaps.
