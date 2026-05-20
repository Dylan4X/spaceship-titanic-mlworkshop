# Experiment Log

## 2026-04-11

### Kaggle notebook references checked

- `gusthema/spaceship-titanic-with-tfdf`
- `samuelcortinhas/spaceship-titanic-a-complete-guide`
- `arunklenin/space-titanic-eda-advanced-feature-engineering`

Local copies:

- `kaggle_notebooks/gusthema_tfdf/spaceship-titanic-with-tfdf.ipynb`
- `kaggle_notebooks/samuel_guide/spaceship-titanic-a-complete-guide.ipynb`
- `kaggle_notebooks/arunklenin_advanced/space-titanic-eda-advanced-feature-engineering.ipynb`

### CV protocol

- Search CV: repeated `StratifiedGroupKFold`, seeds `7, 21, 42, 87, 123`
- Audit CV: repeated `StratifiedGroupKFold`, seeds `3, 11, 17, 29, 57, 68, 99, 131`
- Group key: `PassengerId` group prefix
- Promote only if a candidate improves both search and audit OOF by at least `0.001`
- Threshold stays fixed at `0.5`

### Feature ablation from notebook ideas

Baseline `lgb_a`:

- Search OOF: `0.81203`
- Audit OOF: `0.81226`

Tried:

- `combined_counts`: no gain
- `FamilySize`: search worse, audit better; not promoted
- `CabinRegion`: worse; not promoted
- `AgeGroupFine`: search `0.81318`, audit `0.81330`; promoted for `LGB` single-model only
- Combined notebook-style bundle: worse; not promoted

### Blend audit

Current promoted blend:

- `LGBM 0.70 + XGBoost 0.30`
- Search OOF: `0.81307`
- Audit OOF: `0.81341`

Weight search:

- `0.70 / 0.30` remains the only weight that passes both search and audit promotion checks.

`AgeGroupFine` on top of the blend:

- Base blend search OOF: `0.81307`
- Base blend audit OOF: `0.81341`
- Age-fine blend search OOF: `0.81272`
- Age-fine blend audit OOF: `0.81226`
- Not promoted.

### Current submission decision

Keep current promoted candidate:

- `submission_blend.csv`

Do not replace it with the age-fine version.

## 2026-04-12

### Current public leaderboard anchor

- Best clean submission remains `submission_catboost_pycaret_exact.csv`.
- Public score: `0.81061`.
- Same public score: `submission_catboost_pycaret_count2230.csv`, `submission_catboost_pycaret_exact_scaled.csv`.

### Confirmed failures after the `0.81061` anchor

- `submission_pycaret_catboost_bootstrap100_refcount.csv`
  - Idea: 100-model bootstrap median around the PyCaret/CatBoost exact model.
  - OOB looked strong (`~0.8157`) and changed only 70 rows vs exact.
  - Public score: `0.80593`.
  - Conclusion: OOB/bootstrap stability is misleading here.

- `submission_samet_catboost_081248_repro.csv`
  - Idea: reproduce the `sametkrcan` CatBoost `0.81248` notebook path.
  - Public score: `0.80944`.
  - Conclusion: the local reproduction does not beat the PyCaret exact anchor.

- `submission_catboost_pycaret_exact_scaled.csv`
  - Idea: add the PyCaret notebook's robust-normalization detail.
  - Changed only two rows vs exact.
  - Public score: `0.81061`.
  - Conclusion: tied with exact; no improvement.

- `submission_blend_exact_bsthere_rank_w02.csv`
  - Idea: 2% rank blend of PyCaret/CatBoost exact with a clean self-contained ensemble from a `0.82137` notebook, preserving exact positive count.
  - Changed only six rows vs exact.
  - Public score: `0.81014`.
  - Conclusion: even tiny rank perturbations can hurt; exact is a sharp local optimum.

### Notebook review notes

- `bsthere/spaceship-titanic-0-82137-solution` and `shivanshcoding/0-821-solution-spaceship-titanic-using-lr` contain embedded `BEST_PUBLIC_OVERRIDE_BITS`.
  - These are excluded as non-clean leaderboard/public-label overrides.
  - The clean self-contained ensemble portion was tested separately and did not improve when blended conservatively.

- Exponential weighted ensemble notebooks read many external submission CSVs and blend by public score.
  - Excluded because this is leaderboard submission blending rather than a clean model from train/test.

- `eu1234/spaceship-81-1-leaderboard-top-2-step-by-step` reproduction was unstable locally.
  - Full-feature and `final_features` variants had poor/unreliable CV behavior and large divergence vs exact.
  - Not submitted.

- `guanlintao/0-814-optuna-xgb-space-titanic` reproduction produced low local CV and a very different test distribution.
  - Not submitted.

### Practical rule after 2026-04-12

- Keep `submission_catboost_pycaret_exact.csv` as the clean best deliverable.
- Do not trust OOB/CV improvements unless the resulting test predictions stay extremely close to exact and there is a strong reason beyond CV.
- Do not use notebooks that embed public override bitstrings or external submission blends.

### Additional 2026-04-12 search after broad notebook sweep

- `flaykaer/top-20-code-0-81201-leadeboard`
  - Clean CatBoost + GroupKFold notebook; reproduced as `submission_flaykaer_catboost_exact_repro.csv`.
  - Local GroupKFold accuracy: `0.81422`; diff vs PyCaret exact: `331` rows.
  - Public score: `0.80991`.
  - Conclusion: clean and interpretable, but does not beat the PyCaret exact anchor.

- PyCaret exact + flaykaer probability blend with group smoothing
  - Candidate: `submission_ens_pycaret_flay_exact_a35_groupsmooth_w040.csv`.
  - Local GroupKFold OOF accuracy looked strong: `0.81870`.
  - Public score: `0.80851`.
  - Conclusion: group smoothing overfits local CV and is not a reliable public-LB improvement.

- OOF-mined interpretable rule fix
  - Candidate: `submission_pycaret_rulefix_top1.csv`.
  - Rule: force `Transported=True` when `Destination=PSO J318.5-22` and no luxury spend.
  - Changed only `13` rows vs PyCaret exact.
  - Public score: `0.80991`.
  - Conclusion: even small, interpretable OOF-positive flips can hurt public; exact is a sharp local optimum.

- User-noted early `submission (3).csv`
  - Sanity check submission public score: `0.80640`.
  - Conclusion: not the current anchor; keep PyCaret exact at `0.81061`.

- `defcodeking/spaceship-titanic-prepared-datasets`
  - Downloaded and evaluated prepared feature matrices with CatBoost and LGBM.
  - Best local prepared candidate: `train_prepared_groupid_le.csv` + CatBoost default, OOF `0.81537`.
  - Test distribution was badly shifted: about `1838` positives and `518+` row diff vs exact.
  - Conclusion: do not submit; local prepared CV is not aligned with public.

- Extra notebooks checked or pulled in this round:
  - `travin5/random-forest-with-accuracy-0-816`: single holdout RandomForest, weak feature processing.
  - `mtpc4s9/light-gbm-tuning-with-optuna-auc-rog-0-8181`: single split LGBM threshold tuning, not LB-proof.
  - `shadesh/inf2008-project-galactic-explorer`: CatBoost/feature-selection workflow but heavy random imputation; not a clean next submission.
  - `fathurwithyou/0-81225-spaceship-titanic-neural-network`: no credible local CV, train-loss early stopping, one-hot alignment risk.
  - `ishanpurohit/top-5-solution-with-detailed-explanation`: external pickle was only transparent CatBoost params, but rewritten self-contained version had weak GroupKFold (`0.80743`).

### Updated practical rule

- Treat local CV as a coarse filter only; for this competition it is not a dependable ranker near the `0.8106` public-LB region.
- Avoid further local tweaks around PyCaret exact unless they have a genuinely different modeling assumption; small flips and group smoothing have repeatedly failed.
- Current clean best deliverable remains `submission_catboost_pycaret_exact.csv` / `submission_catboost_pycaret_exact_scaled.csv` / `submission_catboost_pycaret_count2230.csv` at public `0.81061`.

### Final two-submission push

- `submission (3).csv`
  - User had noted this as an early high candidate; resubmitted as sanity check.
  - Public score: `0.80640`.
  - Conclusion: not the current anchor.

- `opamusora/top-10-notebook`
  - Original notebook had an invalid local-CV detail: train numeric missing values were filled by `Transported` group mean.
  - Reworked a clean version with no target-impute and `Cabin` missing set to `Missing/-1/Missing`.
  - Clean GroupKFold OOF: `0.81905`; threshold `0.5` was also the best OOF threshold.
  - Submitted fold-averaged test probabilities with `2230` positives: `submission_opamusora_clean_missing_foldavg_th050.csv`.
  - Public score: `0.80593`.
  - Conclusion: even a genuinely different, high-CV CatBoost candidate is not aligned with public LB.

### Current stopping point

- No clean candidate found above public `0.81061`.
- Strongest clean deliverable remains `submission_catboost_pycaret_exact.csv`.
- If more attempts become available, next work should not rely on local CV rank alone; it should either:
  - run a fundamentally different AutoML stack in a compatible environment, or
  - find a clean notebook with reproducible public-LB evidence and no public override/external submission blending.
