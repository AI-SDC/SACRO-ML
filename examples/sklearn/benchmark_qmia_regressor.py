"""Benchmark: QMIA (HistGradientBoostingRegressor) vs WorstCase vs LiRA.

Compares QMIA against existing MIA attacks across datasets of increasing
size and complexity.  Reports AUC, TPR, FPR, Advantage, and wall time.

Usage:
    .venv/bin/python examples/sklearn/benchmark_qmia_regressor.py
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sacroml.attacks.likelihood_attack import LIRAAttack
from sacroml.attacks.qmia_attack import QMIAAttack
from sacroml.attacks.target import Target
from sacroml.attacks.worst_case_attack import WorstCaseAttack

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")


def _make_target(x, y, name):
    """Build a Target from feature/label arrays."""
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.4, stratify=y, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_tr, y_tr)
    target = Target(
        model=model,
        dataset_name=name,
        X_train=x_tr,
        y_train=y_tr,
        X_test=x_te,
        y_test=y_te,
        X_train_orig=x_tr,
        y_train_orig=y_tr,
        X_test_orig=x_te,
        y_test_orig=y_te,
    )
    for i in range(x.shape[1]):
        target.add_feature(f"V{i}", [i], "float")
    return target


def _run(cls, tgt, **kw):
    """Run a single attack, return (metrics_dict, elapsed_seconds)."""
    d = tempfile.mkdtemp()
    try:
        obj = cls(output_dir=d, write_report=False, **kw)
        t0 = time.perf_counter()
        out = obj.attack(tgt)
        elapsed = time.perf_counter() - t0
        if not out:
            return None, elapsed
        m = out["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]
        return m, elapsed
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _build_scenarios():
    """Generate benchmark scenarios of increasing size and complexity."""
    scenarios = []

    # Real dataset
    bc_x, bc_y = load_breast_cancer(return_X_y=True)
    scenarios.append(("Breast Cancer (569x30)", bc_x, bc_y))

    # Binary synthetic - escalating size
    for n, d, sep, label in [
        (200, 8, 1.5, "tiny"),
        (500, 10, 1.25, "small"),
        (2_000, 20, 1.0, "medium"),
        (5_000, 30, 0.8, "large"),
        (10_000, 50, 0.6, "xlarge"),
        (50_000, 50, 0.5, "xxlarge"),
    ]:
        x, y = make_classification(
            n_samples=n,
            n_features=d,
            n_informative=d // 2,
            n_redundant=0,
            n_classes=2,
            n_clusters_per_class=1,
            class_sep=sep,
            random_state=42,
        )
        scenarios.append((f"{label} (n={n}, d={d})", x, y))

    # Multiclass synthetic
    for n, d, c, sep, label in [
        (500, 10, 3, 1.5, "multi_small"),
        (5_000, 30, 5, 0.8, "multi_large"),
        (20_000, 50, 10, 0.5, "multi_xlarge"),
    ]:
        x, y = make_classification(
            n_samples=n,
            n_features=d,
            n_informative=d // 2,
            n_redundant=0,
            n_classes=c,
            n_clusters_per_class=1,
            class_sep=sep,
            random_state=42,
        )
        scenarios.append((f"{label} (n={n}, d={d}, C={c})", x, y))

    return scenarios


def _v(val):
    """Format a metric value (missing or NaN values render as "—")."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.3f}"


def _vt(val):
    """Format a wall-time value in seconds (missing values render as "—")."""
    if val is None:
        return "—"
    return f"{val:.2f}s"


def _run_all():
    """Run all attacks across all scenarios."""
    scenarios = _build_scenarios()
    results = {}

    for sname, feat, lab in scenarios:
        tgt = _make_target(feat, lab, sname[:20])
        nc = len(np.unique(lab))
        n = feat.shape[0]
        print(f"  {sname} ...", flush=True)

        # QMIA always runs
        cfgs = [("QMIA", QMIAAttack, {})]

        # WorstCase always runs
        cfgs.append(("WorstCase", WorstCaseAttack, {"n_reps": 3}))

        # LiRA only for binary and capped by dataset size
        if nc == 2:
            for ns in [10, 50]:
                if n <= max(5000, ns * 100):
                    cfgs.append((f"LiRA-{ns}", LIRAAttack, {"n_shadow_models": ns}))

        for aname, acls, akw in cfgs:
            m, t = _run(acls, tgt, **akw)
            if m:
                results[(sname, aname)] = {
                    "auc": round(m["AUC"], 3),
                    "tpr": round(m["TPR"], 3),
                    "fpr": round(m["FPR"], 3),
                    "adv": round(m.get("Advantage", abs(m["TPR"] - m["FPR"])), 3),
                    "tpr@0.1": m.get("TPR@0.1", float("nan")),
                    "tpr@0.01": m.get("TPR@0.01", float("nan")),
                    "tpr@0.001": m.get("TPR@0.001", float("nan")),
                    "time": round(t, 2),
                }

    return results


def _g(results, sn, an, field):
    """Look up ``field`` for the ``(scenario, attack)`` pair, or ``None``."""
    r = results.get((sn, an))
    return r[field] if r else None


def _print_section(results, sns, attacks, title, field, fmt_fn=_v):
    """Print a single comparison table section."""
    print(f"\n\n### {title}\n")
    h = f"{'Dataset':<35} {'QMIA':>8} {'Worst':>8} {'LiRA-10':>8} {'LiRA-50':>8}"
    print(h)
    print("-" * len(h))
    for s in sns:
        vals = [fmt_fn(_g(results, s, a, field)) for a in attacks]
        print(f"  {s:<33} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8}")


def _print_tables(results):
    """Print formatted comparison tables."""
    sns = list(dict.fromkeys(k[0] for k in results))
    attacks = ["QMIA", "WorstCase", "LiRA-10", "LiRA-50"]

    _print_section(results, sns, attacks, "AUC Comparison", "auc")
    for fpr_key, label in [
        ("tpr@0.1", "TPR @ FPR=0.1 (higher = better)"),
        ("tpr@0.01", "TPR @ FPR=0.01 (higher = better)"),
        ("tpr@0.001", "TPR @ FPR=0.001 (higher = better)"),
    ]:
        _print_section(results, sns, attacks, label, fpr_key)
    _print_section(
        results, sns, attacks, "FPR at default threshold (lower = better)", "fpr"
    )
    _print_section(results, sns, attacks, "Speed (seconds)", "time", fmt_fn=_vt)

    # Full detail
    print("\n\n### Full Results\n")
    h = (
        f"{'Dataset':<35} {'Attack':<10}"
        f" {'Time':>7} {'AUC':>6}"
        f" {'TPR@.1':>7} {'TPR@.01':>8} {'TPR@.001':>9}"
        f" {'FPR':>6}"
    )
    print(h)
    print("-" * len(h))
    for s in sns:
        for a in attacks:
            r = results.get((s, a))
            if r:
                print(
                    f"  {s:<33} {a:<10}"
                    f" {_vt(r['time']):>7} {_v(r['auc']):>6}"
                    f" {_v(r['tpr@0.1']):>7} {_v(r['tpr@0.01']):>8}"
                    f" {_v(r['tpr@0.001']):>9}"
                    f" {_v(r['fpr']):>6}"
                )
        print()

    # Summary totals
    for a in attacks:
        total = sum(r["time"] for (s, b), r in results.items() if b == a)
        if total > 0:
            print(f"  {a:<10} total: {total:.1f}s")


if __name__ == "__main__":
    print("QMIA Benchmark: QMIA (HGBT) vs WorstCase vs LiRA\n")
    _print_tables(_run_all())
