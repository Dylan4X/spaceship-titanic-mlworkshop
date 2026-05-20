# Spaceship Titanic UpScore Plan

## Current baseline

- Validation scheme: repeated `StratifiedGroupKFold`
- Split key: `PassengerId` group prefix
- Seeds: `7, 21, 42, 87, 123`
- Folds: `5`
- Current locked baseline: `LightGBM + extended features + lgb_a params`
- Current repeated-group CV mean: about `0.8103`
- Current repeated-group OOF at threshold `0.5`: about `0.8120`
- Why this split:
  - `44.7%` of rows belong to multi-passenger groups
  - `87.2%` of groups are label-homogeneous
  - plain `StratifiedKFold` leaks group identity into validation

## What already improved locally

Measured first on a smaller repeated group CV ablation:

- Base LightGBM features:
  - seed-mean CV about `0.8090`
  - OOF about `0.8108`
- Extended feature set:
  - seed-mean CV about `0.8104`
  - OOF about `0.8130`
- XGBoost with the same extended features:
  - seed-mean CV about `0.8098`
  - OOF about `0.8109`

Conclusion:

- The next stable baseline should be `LightGBM + extended features`
- `XGBoost` is useful as an ensemble candidate, but it is not better alone
- The safer single-model setup is `lgb_a`
- `lgb_b` looked slightly better on a short search, but the gap disappeared on a wider seed check

## Extended features now worth keeping

- `AgeMissing`
- `CabinKnown`
- `CabinDeckSide`
- `Route` (`HomePlanet + Destination`)
- `SpendBins`
- `NonZeroSpendCount`
- `LogTotalSpend`
- `CabinNumBin`
- `YoungCryo`

These were chosen because they add structure without using target leakage.

## Things tested and rejected

- Group-level hard/mean voting on validation groups
  - Local CV dropped sharply to around `0.75`
  - Not a safe next step
- Promoting `lgb_b` over `lgb_a` based on a tiny local gain
  - Wider repeated-seed check showed both models are effectively tied
  - `lgb_b` beating `lgb_a` was not robust enough to trust
- OOF threshold micro-tuning
  - Small threshold shifts looked better locally but are easy to overfit
  - The safer default is `0.5`

## Recommended next experiments

1. Lock the new LightGBM baseline and regenerate OOF
2. Add a reproducible model-comparison runner
3. Tune LightGBM on repeated group CV
4. Train XGBoost on the same folds and blend probabilities
5. Only after model choice is stable, tune threshold on OOF

## Tuning priorities

For LightGBM:

- `num_leaves`
- `min_child_samples`
- `learning_rate`
- `n_estimators`
- `subsample`
- `colsample_bytree`
- `reg_alpha`
- `reg_lambda`

For XGBoost:

- `max_depth`
- `min_child_weight`
- `subsample`
- `colsample_bytree`
- `reg_alpha`
- `reg_lambda`

## Decision rule for future experiments

- Keep a change only if it improves repeated group CV mean
- Prefer changes that also improve OOF, not just one lucky seed
- If the gain is only around `0.0005` to `0.0010`, treat it as suspicious until it survives a wider seed check
- Treat threshold tuning as the last mile, not the main source of gains
