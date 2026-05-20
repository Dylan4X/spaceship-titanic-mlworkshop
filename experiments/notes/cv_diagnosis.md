# CV Diagnosis

## What went wrong

The local CV scheme itself was not fundamentally broken.

The real issue was model selection discipline:

- `lgb_a` and `lgb_b` were extremely close
- the apparent gain from `lgb_b` came from a short repeated-seed run plus threshold tuning
- that gain did not survive a wider repeated-seed check

## Wider check result

Using a wider repeated `StratifiedGroupKFold` check over more seeds:

- `lgb_a`
  - fold mean about `0.8108`
  - OOF at threshold `0.5` about `0.8125`
- `lgb_b`
  - fold mean about `0.8112`
  - OOF at threshold `0.5` about `0.8125`

Conclusion:

- the two models are effectively tied under the current local CV
- choosing `lgb_b` over `lgb_a` was not justified strongly enough

## Practical rule going forward

- Do not trust gains smaller than about `0.001` unless they survive a wider seed check
- Avoid changing threshold away from `0.5` unless the gain is consistent across multiple reruns
- Prefer the simpler, already proven model when the difference is within noise
