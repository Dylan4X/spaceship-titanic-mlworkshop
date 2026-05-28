# Experiment Evidence

This folder keeps the compact evidence tables used by the final report.

The report's main experimental coverage is:

- Unified benchmark across Logistic Regression, Random Forest, LightGBM, XGBoost, and CatBoost.
- Feature-family ablation on the clean CatBoost line.
- Hyperparameter tuning evidence for LightGBM, CatBoost, and the compact XGBoost branch.
- CV-versus-public leaderboard comparison.
- Stage-wise reconstruction of the public-best XGBoost branch.

The final public-best XGBoost code is in `code/reproduce_081716_xgb.py` and `notebooks/demo_081716_xgb.ipynb`.

To rerun a small local benchmark and feature ablation check:

```powershell
python code\run_local_experiments.py
```
