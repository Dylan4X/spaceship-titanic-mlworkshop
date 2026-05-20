# CV Protocol

## Goal

Build a local validation process that is strict enough to guide model selection without rewarding tiny, noisy gains.

## Split design

- Base splitter: repeated `StratifiedGroupKFold`
- Group key: `PassengerId` prefix
- Fold count: `5`

Why:

- many passengers belong to shared groups
- group members often behave similarly
- plain row-level CV leaks group structure into validation

## Search vs audit

Do not use one seed set for everything.

- Search seeds:
  - `7, 21, 42, 87, 123`
  - used for feature and hyperparameter exploration
- Audit seeds:
  - `3, 11, 17, 29, 57, 68, 99, 131`
  - used only to decide whether a candidate is real enough to promote

## Promotion rule

Default baseline:

- `lgb_a`
- threshold fixed at `0.5`

A candidate should be promoted only if:

- it beats baseline on search OOF by at least `0.001`
- and it also beats baseline on audit OOF by at least `0.001`

If one side fails, treat the candidate as noise.

## Guardrails

- Do not tune threshold on audit seeds
- Do not accept `0.000x` gains as meaningful
- Prefer simpler models when the difference is inside noise
- Keep leaderboard checks as a sanity check, not the primary optimizer
