# MetaAttack Implementation Plan

**Spec:** `docs/specs/2026-04-12-meta-attack-design.md`  
**Branch:** `428-meta-attack`

---

## Stage 1 — Module skeleton and factory registration

**Files:**
- `sacroml/attacks/meta_attack.py` (NEW)
- `sacroml/attacks/factory.py` (EDIT)

**Work:**
- Create `MetaAttack(Attack)` with full `__init__`:
  - Parse and validate `attacks` list via `_parse_attacks()`
  - Load `k_threshold` from ACRO if None
  - Store `mia_threshold`, `k_threshold`, `vulnerability_df`
- Implement `_parse_attacks()`: normalise 2-tuple to 3-tuple, validate names against `SUPPORTED_ATTACKS`
- Stub abstract methods: `attackable`, `_attack`, `_get_attack_metrics_instances`, `_make_pdf`, `__str__`
- `attackable` fully implemented: `target.has_model() and target.has_data()`
- `__str__` returns `"Meta Attack"`
- Add `"meta": MetaAttack` to factory registry
- Use lazy import in factory to avoid load-time issues: `from sacroml.attacks.meta_attack import MetaAttack`

**Commit:** `feat: add MetaAttack class skeleton and factory registration`

**Verify:** `python -c "from sacroml.attacks.meta_attack import MetaAttack"` imports without error. `MetaAttack(attacks=[("lira", {})]).attack_specs` returns `[("lira", {}, 1)]`. Passing `"worstcase"` raises `ValueError`.

---

## Stage 2 — Sub-attack orchestration

**Files:**
- `sacroml/attacks/meta_attack.py` (EDIT)

**Work:**
- Implement `_run_sub_attack(name, params, target, run_idx) -> Attack`:
  - Use lazy import: `from sacroml.attacks.factory import create_attack`
  - Build sub_output_dir: `{self.output_dir}/{name}_run{run_idx}`
  - Inject `report_individual=True` for MIA attacks (lira, qmia)
  - Inject `output_dir=sub_output_dir`, `write_report=False`
  - Create and run the sub-attack, return the attack object
- Implement orchestration loop in `_attack()`:
  - For each `(name, params, n_reps)` in `self.attack_specs`:
    - For each `rep` in `range(n_reps)`:
      - Call `_run_sub_attack(name, params, target, rep)`
      - Store the returned attack object
  - (Score extraction and DataFrame construction delegated to later stages — store raw attack objects for now)

**Commit:** `feat: implement sub-attack orchestration in MetaAttack`

**Verify:** Create a MetaAttack with `[("structural", {})]`, call `_attack(target)` on a test target. Verify `structural_run0/` subdirectory is created.

---

## Stage 3 — Score extraction

**Files:**
- `sacroml/attacks/meta_attack.py` (EDIT)

**Work:**
- Implement `_extract_mia_scores(attack_obj, name) -> list[float]`:
  - LiRA: return `attack_obj.attack_metrics[-1]["individual"]["score"]`
  - QMIA: return `attack_obj.attack_metrics[0]["individual"]["member_prob"]`
  - Raise `RuntimeError` if individual data missing (attack didn't set report_individual)
- Implement `_extract_structural_scores(attack_obj) -> dict`:
  - Return `{ "k_anonymity": attack_obj.record_level_results.k_anonymity, "class_disclosure": attack_obj.record_level_results.class_disclosure, "smallgroup_risk": attack_obj.record_level_results.smallgroup_risk }`
  - Access dataclass directly (always populated regardless of report_individual)
- Wire extraction into `_attack()` orchestration loop:
  - After each sub-attack run, extract scores immediately
  - Store as `{name: [list_of_scores_per_rep]}` dict

**Commit:** `feat: implement per-record score extraction from sub-attacks`

**Verify:** Run MetaAttack with a single LiRA (n_shadow_models=2) on a test target. Verify extracted scores are a list of floats with length = n_train + n_test.

---

## Stage 4 — DataFrame construction

**Files:**
- `sacroml/attacks/meta_attack.py` (EDIT)

**Work:**
- Implement `_build_dataframe(target, mia_scores, structural_scores) -> pd.DataFrame`:
  - Create `is_member` column: `[1]*n_train + [0]*n_test`
  - **Level 1 — within-attack aggregation** (for each MIA attack):
    - Stack per-rep score lists into numpy array (shape: n_reps x n_records)
    - Compute `{name}_mean = np.mean(axis=0)`
    - Compute `{name}_std = np.std(axis=0)`
    - Compute `{name}_consistency = np.mean(scores > self.mia_threshold, axis=0)`
    - Compute `{name}_vuln = {name}_mean > self.mia_threshold`
  - **Structural columns** (for each structural rep):
    - Stack reps, compute mean k-anonymity and vuln consistency
    - For single rep: direct assignment
    - Pad with NaN to n_train + n_test (structural is train-only)
    - Compute `struct_vuln = (k < k_threshold) | cd | sg`
  - **Level 2 — cross-attack aggregation**:
    - Collect per-attack mean columns for MIA attacks only
    - `mia_mean = np.mean(mia_means, axis=0)` — equal weight per attack
    - `mia_gmean = np.exp(np.mean(np.log(mia_means + eps), axis=0))` — geometric mean with epsilon for numerical stability
    - `n_vulnerable = sum of {name}_vuln columns` (NaN structural doesn't count)
  - Set `self.vulnerability_df = df`
- Wire into `_attack()` after score extraction

**Commit:** `feat: build vulnerability DataFrame with two-level aggregation`

**Verify:** Run MetaAttack with LiRA (n_reps=2) + structural on test target. Verify DataFrame has correct shape (n_train + n_test rows), correct columns, NaN in structural columns for test records, mia_mean in [0,1].

---

## Stage 5 — Global metrics and reporting

**Files:**
- `sacroml/attacks/meta_attack.py` (EDIT)

**Work:**
- Compute global metrics in `_attack()` after DataFrame is built:
  - `mia_means = self.vulnerability_df["mia_mean"].values`
  - `y_pred_proba = np.column_stack([1 - mia_means, mia_means])`
  - `membership = np.array([1]*n_train + [0]*n_test)`
  - `self.attack_metrics = [metrics.get_metrics(y_pred_proba, membership)]`
- Implement `_get_attack_metrics_instances()`:
  - Return standard `{"attack_instance_logger": {"instance_0": {metrics + sub_attacks + individual}}}` structure
  - Include sub-attack summary metrics under `"sub_attacks"` key
  - Include DataFrame as dict-of-lists under `"individual"` key
- Implement `_construct_metadata()`:
  - Call `super()._construct_metadata()`
  - Add meta-specific params (attacks list, thresholds)
- Implement `_make_pdf()`: return `None` for now (PDF generation is a follow-up)
- Wire `_make_report(target)` and `_write_report(output)` into `_attack()` flow
- Also save DataFrame as CSV: `self.vulnerability_df.to_csv({output_dir}/vulnerability_matrix.csv)`

**Commit:** `feat: add global metrics, JSON report, and CSV export`

**Verify:** Run full MetaAttack. Check `report.json` has correct structure. Check `vulnerability_matrix.csv` exists and is loadable. Check global AUC is in [0,1].

---

## Stage 6 — Test suite

**Files:**
- `tests/attacks/test_meta_attack.py` (NEW)

**Work:**
- Fixture: `meta_target` — `make_classification(n_samples=200, n_features=8)`, RandomForest(n_estimators=50), Target with features registered
- Test cases:
  1. `test_meta_basic` — LiRA(n_shadow_models=2) + structural, verify DataFrame shape = (200, expected_cols)
  2. `test_meta_repeated_runs` — LiRA(n_shadow_models=2, n_reps=2), verify std column > 0 for at least some records, consistency in [0,1]
  3. `test_meta_unsupported_attack` — `("worstcase", {})` raises ValueError
  4. `test_meta_invalid_tuple` — `("lira",)` raises ValueError
  5. `test_meta_threshold_effects` — run with threshold=0.3 vs 0.7, verify different vuln counts
  6. `test_meta_test_record_nan` — structural columns NaN for records where is_member=0
  7. `test_meta_global_metrics` — AUC, TPR in [0,1]
  8. `test_meta_report_structure` — output dict has correct nested keys
  9. `test_meta_factory_integration` — `factory.attack(target, "meta", attacks=[...])` works
  10. `test_meta_dataframe_property` — `meta.vulnerability_df` is a pd.DataFrame with expected dtypes

**Commit:** `test: add MetaAttack test suite`

**Verify:** `pytest tests/attacks/test_meta_attack.py -v` all pass.

---

## Stage 7 — Example script

**Files:**
- `examples/sklearn/meta_attack_example.py` (NEW)

**Work:**
- Load an existing trained target (reuse cancer or nursery example's target dir)
- Run MetaAttack with LiRA + QMIA + structural
- Print DataFrame summary statistics
- Show which records are vulnerable to all attacks
- Show top-10 most vulnerable records by n_vulnerable and mia_mean

**Commit:** `docs: add MetaAttack example script`

**Verify:** Script runs end-to-end and prints meaningful output.

---

## Commit Summary

| Stage | Commit message | Depends on |
|-------|---------------|------------|
| 1 | `feat: add MetaAttack class skeleton and factory registration` | — |
| 2 | `feat: implement sub-attack orchestration in MetaAttack` | Stage 1 |
| 3 | `feat: implement per-record score extraction from sub-attacks` | Stage 2 |
| 4 | `feat: build vulnerability DataFrame with two-level aggregation` | Stage 3 |
| 5 | `feat: add global metrics, JSON report, and CSV export` | Stage 4 |
| 6 | `test: add MetaAttack test suite` | Stage 5 |
| 7 | `docs: add MetaAttack example script` | Stage 5 |

Stages 6 and 7 are independent of each other and can be done in either order.
