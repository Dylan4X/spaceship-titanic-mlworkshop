# Experiment Audit Against the Final Paper

## Verdict

The current experiment evidence is sufficient for the final paper and course rubric. I do not recommend adding a large new tuning campaign before submission because the paper's central claim is not "Optuna found a uniquely optimal model"; it is "validation-first evidence and public leaderboard behavior diverged, so the clean CatBoost anchor and public-best XGBoost branch should be reported separately."

The highest-score XGBoost branch should still be presented as evidence-backed rather than arbitrary. The right framing is:

- It is not the best strict local-CV model.
- It is the best public-leaderboard branch.
- It has a documented provenance: original Optuna-labeled notebook parameters, standalone reproduction, mini Optuna rerun, focused neighborhood search, stage reconstruction, and public submission milestones.
- This supports a credible competition narrative without pretending that local CV fully explains the leaderboard result.

## Tuning Evidence Already Present

- `tuning_summary.csv`: clean LightGBM and CatBoost tuning gains.
- `tuning_key_changes.csv`: concrete before/after hyperparameter changes for LightGBM and CatBoost.
- `xgb_neighborhood_search.csv`: focused neighborhood search around the public-best XGBoost parameters.
- `xgb_branch_reconstruction.csv`: stage-wise reconstruction of compact preprocessing, outlier diagnostic, feature pruning, and strict group-aware audit.
- `pipeline_audit.csv`: public-vs-local comparison for representative candidate lines.
- `xgb_parameter_provenance.csv`: compact evidence map showing why the public-best XGB branch is not a one-off guess.

## What Was Added for Demo Readiness

- `code/reproduce_highscore_xgb.py`: command-line reproduction of the high-score XGBoost branch.
- `notebooks/demo_repro_optuna_xgb_space_titanic.ipynb`: cleaned demo notebook.
- The demo path now assumes `optuna` is installed and runs a small Optuna search by default.

## Remaining Non-Code Placeholders

- `report/final_ieee_paper.tex` still contains a placeholder for the final GitHub URL.
- `report/final_ieee_paper.tex` still contains a placeholder for official student names/IDs and member-level contribution mapping.

Those are administrative facts, not modeling gaps.
