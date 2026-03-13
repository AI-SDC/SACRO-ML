# QMIA Tuning Log

## Purpose

Track whether `QMIAAttack` is working, how it compares to shadow-model attacks in
runtime and attack quality, and which knobs are most worth tuning.

This log reflects the current binary tabular v1 implementation in
`sacroml/attacks/qmia_attack.py`.

## Current Implementation Scope

- Binary classification only.
- Processed tabular features only.
- Public non-member calibration slice is `target.X_test`.
- Default QMIA mode uses CatBoost `RMSEWithUncertainty`.
- Fallback mode uses direct CatBoost quantile regression.

## What Is Working

- `QMIAAttack` runs end to end and integrates with the attack factory.
- It returns `{}` for unsupported targets.
- It emits standard `sacroml.metrics.get_metrics()` outputs.
- It supports both:
  - `use_gaussian=True`
  - `use_gaussian=False`
- The focused QMIA test suite passed:
  - `13 passed, 1 deselected`

## Benchmark Setup

Compared:

- `qmia_gaussian`
- `qmia_quantile`
- `lira_20`
- `lira_40`

Target model:

- `RandomForestClassifier(n_estimators=50)`

Datasets:

1. `small_easy`
   - `n_samples=240`
   - `n_features=8`
   - `class_sep=1.25`
2. `medium_harder`
   - `n_samples=600`
   - `n_features=16`
   - `class_sep=0.9`

QMIA parameters:

- `catboost_params={"iterations": 20, "depth": 3}`

## Benchmark Results

### `small_easy`

| Attack | Time (s) | AUC | Advantage | TPR | FPR | Public FPR |
|---|---:|---:|---:|---:|---:|---:|
| `qmia_gaussian` | 0.2486 | 0.5405 | 0.0799 | 0.1111 | 0.0313 | 0.0313 |
| `qmia_quantile` | 0.0811 | 0.6001 | 0.0694 | 0.1111 | 0.0417 | 0.0417 |
| `lira_20` | 0.8268 | 0.7920 | 0.4167 | 0.8125 | 0.3958 | n/a |
| `lira_40` | 1.5092 | 0.8139 | 0.4271 | 0.7708 | 0.3438 | n/a |

### `medium_harder`

| Attack | Time (s) | AUC | Advantage | TPR | FPR | Public FPR |
|---|---:|---:|---:|---:|---:|---:|
| `qmia_gaussian` | 0.0883 | 0.9150 | 0.2222 | 0.2389 | 0.0167 | 0.0167 |
| `qmia_quantile` | 0.0877 | 0.8944 | 0.0861 | 0.0944 | 0.0083 | 0.0083 |
| `lira_20` | 1.4617 | 0.9460 | 0.5611 | 0.9778 | 0.4167 | n/a |
| `lira_40` | 2.5426 | 0.9570 | 0.5597 | 0.9806 | 0.4208 | n/a |

## Conclusions

### Is QMIA working?

Yes, within the intended v1 scope.

It is:

- functionally integrated,
- producing stable report output,
- controlling public-slice FPR conservatively,
- much cheaper than LiRA in wall-clock runtime.

### Is QMIA computationally better than shadow attacks?

Yes.

In the current sweep, QMIA was consistently much faster than LiRA:

- around `3x` to `18x` faster than LiRA, depending on dataset and shadow count
- with runtime that did not scale with shadow-model count

### Is QMIA stronger than shadow attacks?

Not in the current sweep.

LiRA remained stronger on raw attack effectiveness:

- higher `AUC`
- much higher `TPR`
- much larger attacker `Advantage`

QMIA is currently best viewed as:

- a cheap public-data threshold attack,
- not a drop-in replacement for stronger shadow-based attacks when maximum attack
  strength is the priority.

### What is the main tradeoff?

QMIA:

- much cheaper
- more conservative
- lower false-positive rates
- lower recall in the current configuration

LiRA:

- more expensive
- stronger
- requires shadow models
- higher effective membership separation

## Why QMIA Looks Conservative

The current QMIA implementation:

- fits only on `X_test` as a public non-member slice,
- uses a low default `alpha=0.01`,
- converts `score - threshold` margins into a sigmoid score only to fit the
  existing metrics API,
- does not yet use a richer public calibration dataset or feature engineering.

This tends to produce:

- low public FPR,
- modest or low TPR,
- decent ranking quality on some datasets,
- weaker absolute attack power than LiRA.

## Best Knobs To Tune First

### 1. `alpha`

Most important operational knob.

Effects:

- higher `alpha` lowers the threshold
- usually increases TPR
- usually increases FPR
- changes the attack's operating point more than its ranking behavior

Recommendation:

- sweep `alpha` over values like:
  - `0.001`
  - `0.005`
  - `0.01`
  - `0.02`
  - `0.05`
  - `0.1`

Use when:

- you want more recall and can tolerate more false positives

### 2. CatBoost capacity

Current benchmark settings were intentionally light:

- `iterations=20`
- `depth=3`

Likely better settings to explore:

- `iterations=100`, `200`, `500`
- `depth=4`, `6`, `8`
- `learning_rate=0.03` to `0.1`
- optional regularization via `l2_leaf_reg`

Effects:

- better threshold fit
- potentially better AUC and TPR
- longer runtime, but still likely far below LiRA

### 3. `use_gaussian`

Compare:

- `True`: uncertainty-regression threshold from `(mu, sigma)`
- `False`: direct quantile regression

Observed in the current sweep:

- quantile mode was sometimes faster
- gaussian mode was sometimes stronger
- neither dominates across all settings

Recommendation:

- keep both modes benchmarked on each target family

### 4. Public calibration data quality

The strongest structural limitation right now is not CatBoost itself, but the
public slice.

Current v1 behavior:

- uses only `target.X_test` / `target.y_test`

Potential improvement:

- use a larger or more representative public non-member dataset
- ensure public distribution matches the target deployment distribution

Expected effect:

- better conditional quantile estimates
- more stable thresholding
- potentially higher attack quality without shadow models

### 5. Feature representation

Current QMIA threshold regressor only sees processed features.

Potential future tweaks:

- add target-score-derived features
- add max probability / entropy / margin features
- add correctness or loss-like derived features

This would be a meaningful change and should be treated as a deliberate v2
experiment, not a silent tweak.

## Recommended Tuning Order

1. Sweep `alpha`.
2. Compare `use_gaussian=True` vs `False`.
3. Increase CatBoost capacity.
4. Benchmark against LiRA at matched runtime budgets.
5. Improve the public calibration slice.

## Practical Benchmarking Guidance

When tuning QMIA, track at least:

- runtime in seconds
- `AUC`
- `Advantage`
- `TPR`
- `FPR`
- `observed_public_fpr`

Recommended comparison baselines:

- `QMIAAttack(use_gaussian=True)`
- `QMIAAttack(use_gaussian=False)`
- `LIRAAttack(n_shadow_models=20)`
- `LIRAAttack(n_shadow_models=50 or 100)`

Do not compare only on one metric.

Examples:

- If QMIA is much faster and `AUC` is close, it may be the better practical
  choice.
- If maximum attack power matters, LiRA is still the stronger baseline.

## Reproducible Commands (Documented)

One-command benchmark runs are available via `Makefile`:

- Synthetic development baseline:
  - `make qmia-bench`
- Sklearn preset datasets (second way):
  - `make qmia-bench-sklearn`
- Stronger tuning preset:
  - `make qmia-bench-strong`

Sklearn preset mode currently uses:

- `breast_cancer`
- `wine_binary` (class 0 vs rest; binary staging for QMIA v1)

Example of manual stronger tuning overrides:

`make qmia-bench DATASET_SOURCE=sklearn SKLEARN_DATASETS=breast_cancer,wine_binary LIRA_SHADOW_MODELS=20,40,100 QMIA_ALPHA=0.02 QMIA_ITERATIONS=200 QMIA_DEPTH=6 QMIA_LEARNING_RATE=0.03 QMIA_L2_LEAF_REG=5.0 QMIA_SUBSAMPLE=0.9`

## Current Recommendation

For now:

- use QMIA when computational efficiency and public-data-only thresholding are
  the main goal,
- keep LiRA as the stronger attack baseline for measuring worst-case attack
  strength,
- do not claim QMIA is stronger than shadow-model attacks yet,
- do claim that QMIA is functioning and materially cheaper.

## Next Experiments

- Run an `alpha` sweep and log the TPR/FPR curve.
- Increase CatBoost `iterations` and `depth`.
- Compare Gaussian vs quantile mode on multiple real tabular targets.
- Compare QMIA to LiRA at matched compute budgets rather than only fixed shadow
  counts.
