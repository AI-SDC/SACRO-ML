# MetaAttack Design Spec

**Issue:** #428  
**Branch:** `428-meta-attack`  
**Date:** 2026-04-12  
**Target:** JMLR publication

## Problem

Individual privacy attacks (LiRA, QMIA, Structural) each report per-record vulnerability in isolation. There is no way to:
- Identify records vulnerable to **multiple** attacks simultaneously
- Measure whether vulnerability findings are **consistent** across stochastic runs
- Produce cross-attack analyses (Venn diagrams, correlation, meta-scores)

## Solution

A `MetaAttack` class that orchestrates multiple sub-attacks on the same Target, extracts per-record scores, and aggregates them into a unified pandas DataFrame with two-level aggregation.

## Design Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Input mode | Run attacks internally (option A) | Reproducibility via YAML; forces `report_individual=True` automatically |
| 2 | Repeated runs | Built-in `n_reps` per attack | Two-level aggregation avoids weighting bias; gives confidence intervals |
| 3 | Unsupported attacks | Reject with ValueError | Fail fast; no silent surprises |

## API

```python
from sacroml.attacks.meta_attack import MetaAttack

meta = MetaAttack(
    attacks=[
        ("lira", {"n_shadow_models": 100}, 5),  # (name, params, n_reps)
        ("qmia", {}, 3),                          # n_reps defaults to 1
        ("structural", {}),
    ],
    mia_threshold=0.5,       # Binary flag cutoff for MIA attacks
    k_threshold=None,        # None = ACRO default (10)
    output_dir="outputs",
    write_report=True,
)
output = meta.attack(target)
df = meta.vulnerability_df   # The per-record DataFrame
```

**YAML config equivalent:**
```yaml
attacks:
  - name: meta
    params:
      attacks:
        - ["lira", {"n_shadow_models": 100}, 5]
        - ["qmia", {}, 3]
        - ["structural", {}]
      mia_threshold: 0.5
```

**Supported attacks:** `lira`, `qmia`, `structural` (must have per-record scores).
Passing `worstcase` or `attribute` raises `ValueError` at construction.

## Class Structure

```
MetaAttack(Attack)
    SUPPORTED_ATTACKS = {"lira", "qmia", "structural"}
    MIA_ATTACKS = {"lira", "qmia"}

    __init__(attacks, mia_threshold, k_threshold, output_dir, write_report)
    _parse_attacks(attacks) -> list[tuple[str, dict, int]]
    _attack(target) -> dict
    _run_sub_attack(name, params, target, run_idx) -> Attack
    _extract_mia_scores(attack_obj, name) -> list[float]
    _extract_structural_scores(attack_obj) -> dict
    _build_dataframe(target, all_scores) -> pd.DataFrame
    attackable(cls, target) -> bool
    _get_attack_metrics_instances() -> dict
    _make_pdf(output) -> FPDF | None
    __str__() -> str

    vulnerability_df: pd.DataFrame | None  (property)
```

## Score Extraction Paths (verified)

| Attack | Path after `obj.attack(target)` | Score field | Type | Length |
|--------|-------------------------------|-------------|------|--------|
| LiRA | `obj.attack_metrics[-1]["individual"]` | `"score"` | list[float] [0,1] | n_train + n_test |
| QMIA | `obj.attack_metrics[0]["individual"]` | `"member_prob"` | list[float] [0,1] | n_train + n_test |
| Structural | `obj.record_level_results` | `.k_anonymity`, `.class_disclosure`, `.smallgroup_risk` | list[int], list[bool], list[bool] | n_train only |

**Record ordering:** LiRA and QMIA: train first, then test. Structural: train only.

**Critical detail:** Structural always computes `record_level_results` regardless of `report_individual`. MetaAttack only needs to force `report_individual=True` on MIA attacks.

**Membership ground truth:** Reconstructed from `len(target.X_train)` and `len(target.X_test)` — avoids depending on attack-internal member arrays.

## Two-Level Aggregation

### Level 1: Within-attack (across n_reps)

For each MIA attack with n_reps > 1:
- `{name}_mean`: mean of scores across reps, per record
- `{name}_std`: std of scores across reps, per record
- `{name}_consistency`: fraction of reps where score > mia_threshold, per record

For structural with n_reps > 1:
- `struct_k_mean`, `struct_k_std`: mean/std of k-anonymity across reps
- `struct_vuln_consistency`: fraction of reps where structurally vulnerable

### Level 2: Cross-attack

- `mia_mean`: arithmetic mean of per-attack means (LiRA mean, QMIA mean) — equal weight per attack
- `mia_gmean`: geometric mean of per-attack means
- `n_vulnerable`: count of attacks where record is flagged (structural NaN on test records doesn't count)

### Binary vulnerability flags

- MIA: `{name}_vuln = {name}_mean > mia_threshold`
- Structural: `struct_vuln = (k < k_threshold) OR class_disclosure OR smallgroup_risk`

## DataFrame Schema

For `attacks=[("lira", {}, 3), ("qmia", {}, 2), ("structural", {})]`:

```
Columns:
  is_member          int (1=train, 0=test)
  lira_mean          float [0,1]
  lira_std           float
  lira_consistency   float [0,1]
  lira_vuln          bool
  qmia_mean          float [0,1]
  qmia_std           float
  qmia_consistency   float [0,1]
  qmia_vuln          bool
  struct_k           int (NaN for test)
  struct_cd          bool (NaN for test)
  struct_sg          bool (NaN for test)
  struct_vuln        bool (NaN for test)
  mia_mean           float [0,1]
  mia_gmean          float [0,1]
  n_vulnerable       int

Rows: n_train + n_test (train first, then test)
Index: record_0, record_1, ...
```

When n_reps=1 for a MIA attack, `_std` = 0.0 and `_consistency` = 1.0 or 0.0.

## Sub-Attack Isolation

Each sub-attack gets its own output subdirectory:
```
{output_dir}/
  lira_run0/shadow_models/...
  lira_run1/shadow_models/...
  qmia_run0/...
  structural_run0/...
  report.json    (MetaAttack's own report)
```

## Global Metrics

MetaAttack computes its own global metrics using `mia_mean` as a membership predictor:
```python
y_pred_proba = np.column_stack([1 - mia_means, mia_means])
membership = np.array([1]*n_train + [0]*n_test)
self.attack_metrics = [metrics.get_metrics(y_pred_proba, membership)]
```

This gives the meta-attack its own AUC, TPR, Advantage, etc.

## Report JSON Structure

```json
{
  "log_id": "...",
  "metadata": {
    "attack_name": "Meta Attack",
    "attack_params": {
      "attacks": [...],
      "mia_threshold": 0.5,
      "k_threshold": 10
    },
    "global_metrics": { "AUC": ..., "TPR": ..., ... }
  },
  "attack_experiment_logger": {
    "attack_instance_logger": {
      "instance_0": {
        "AUC": ..., "TPR": ...,
        "sub_attacks": {
          "lira": { "n_reps": 3, "AUC": ... },
          "qmia": { "n_reps": 2, "AUC": ... },
          "structural": { "n_reps": 1 }
        },
        "individual": { ... DataFrame as dict-of-lists ... }
      }
    }
  }
}
```

## Circular Import Avoidance

`factory.py` will import `MetaAttack`. `MetaAttack._attack()` needs `create_attack` from factory. Use lazy import:
```python
def _attack(self, target):
    from sacroml.attacks.factory import create_attack
    ...
```

## Testing Strategy

File: `tests/attacks/test_meta_attack.py`

Fixtures: synthetic data with `make_classification`, trained RandomForest, Target object.
Use small configs for speed: `n_shadow_models=2`, `n_reps=2`.

Test cases:
1. Basic run with 1 LiRA + 1 structural → verify DataFrame shape, column names
2. Repeated runs → verify std > 0 when n_reps > 1, consistency in [0,1]
3. Unsupported attack → ValueError
4. Invalid tuple format → ValueError
5. Threshold effects → changing mia_threshold changes vuln flags
6. Test record NaN → structural columns NaN for non-members
7. Global metrics → AUC in [0,1]
8. JSON report structure → correct keys present
9. Factory integration → `factory.attack(target, "meta", attacks=[...])` works
10. DataFrame export → vulnerability_df is a valid pandas DataFrame

## Files Changed

| File | Change |
|------|--------|
| `sacroml/attacks/meta_attack.py` | **NEW** — MetaAttack class |
| `sacroml/attacks/factory.py` | Add `"meta": MetaAttack` to registry |
| `tests/attacks/test_meta_attack.py` | **NEW** — test suite |
| `examples/sklearn/meta_attack_example.py` | **NEW** — usage example |
