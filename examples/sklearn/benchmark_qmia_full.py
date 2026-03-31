"""Full QMIA benchmark with formatted tables.

Compares QMIA against WorstCase and LiRA across binary, multiclass, real,
and synthetic datasets at multiple scales.

Usage:
    .venv/bin/python examples/sklearn/benchmark_qmia_full.py

Note:
    This script is superseded by ``benchmark_qmia_regressor.py`` which
    includes TPR@FPR comparisons. Kept for backwards compatibility.
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
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.4, stratify=y, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_tr, y_tr)
    target = Target(
        model=model, dataset_name=name,
        X_train=x_tr, y_train=y_tr, X_test=x_te, y_test=y_te,
        X_train_orig=x_tr, y_train_orig=y_tr,
        X_test_orig=x_te, y_test_orig=y_te,
    )
    for i in range(x.shape[1]):
        target.add_feature(f"V{i}", [i], "float")
    return target


def _run(cls, tgt, **kw):
    d = tempfile.mkdtemp()
    try:
        obj = cls(output_dir=d, write_report=False, **kw)
        t0 = time.perf_counter()
        out = obj.attack(tgt)
        elapsed = time.perf_counter() - t0
        if not out:
            return None, elapsed
        m = out["attack_experiment_logger"][
            "attack_instance_logger"
        ]["instance_0"]
        return m, elapsed
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _v(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    return f"{val:.3f}"


def _vt(val):
    if val is None:
        return "-"
    return f"{val:.2f}s"


def _build_scenarios():
    bc_x, bc_y = load_breast_cancer(return_X_y=True)
    scenarios = [("Breast Cancer (569)", bc_x, bc_y)]
    for n, d, c, sep in [
        (500, 10, 2, 1.5), (1000, 20, 2, 1.0),
        (2000, 30, 2, 0.8), (5000, 50, 2, 0.7),
        (10000, 50, 2, 0.5), (20000, 50, 2, 0.4),
        (500, 10, 3, 1.5), (1000, 20, 5, 1.0),
        (2000, 30, 5, 0.8), (5000, 50, 5, 0.6),
        (10000, 50, 10, 0.5),
    ]:
        feat, lab = make_classification(
            n_samples=n, n_features=d, n_informative=d // 2,
            n_redundant=0, n_classes=c, n_clusters_per_class=1,
            class_sep=sep, random_state=42,
        )
        tag = (
            f"n={n}, d={d}, C={c}" if c > 2
            else f"n={n}, d={d}, sep={sep}"
        )
        scenarios.append((tag, feat, lab))
    return scenarios


def _run_all():
    scenarios = _build_scenarios()
    results = {}

    for sname, feat, lab in scenarios:
        tgt = _make_target(feat, lab, sname[:20])
        nc = len(np.unique(lab))
        n = feat.shape[0]

        cfgs = [
            ("QMIA", QMIAAttack, {}),
            ("WorstCase", WorstCaseAttack, {"n_reps": 3}),
        ]
        if nc == 2:
            for ns in [10, 50, 100]:
                if n <= max(5000, ns * 100):
                    cfgs.append((
                        f"LiRA-{ns}", LIRAAttack,
                        {"n_shadow_models": ns},
                    ))

        for aname, acls, akw in cfgs:
            m, t = _run(acls, tgt, **akw)
            if m:
                results[(sname, aname)] = {
                    "auc": round(m["AUC"], 3),
                    "tpr": round(m["TPR"], 3),
                    "fpr": round(m["FPR"], 3),
                    "adv": round(
                        m.get("Advantage", abs(m["TPR"] - m["FPR"])),
                        3,
                    ),
                    "tpr1": m.get("TPR@0.01", float("nan")),
                    "tpr01": m.get("TPR@0.001", float("nan")),
                    "pfpr": m.get(
                        "observed_public_fpr", float("nan")
                    ),
                    "time": round(t, 2),
                }

    return results


def _g(results, sn, an, field):
    r = results.get((sn, an))
    return r[field] if r else None


def _print_tables(results):
    sns = list(dict.fromkeys(k[0] for k in results))
    real = [s for s in sns if not s.startswith("n=")]
    binary = [s for s in sns if s.startswith("n=") and "C=" not in s]
    multi = [s for s in sns if "C=" in s]
    attacks = ["QMIA", "WorstCase", "LiRA-10", "LiRA-50", "LiRA-100"]

    # AUC
    print("\n### AUC Comparison\n")
    h = (
        f"{'Dataset':<28} {'QMIA':>9} {'WorstCase':>10}"
        f" {'LiRA-10':>9} {'LiRA-50':>9} {'LiRA-100':>9}"
    )
    print(h)
    print("\u2500" * len(h))
    for label, grp in [
        ("REAL DATASETS", real),
        ("BINARY \u2014 SYNTHETIC", binary),
        ("MULTICLASS", multi),
    ]:
        if not grp:
            continue
        print(f"  {label}")
        for s in grp:
            vals = [_v(_g(results, s, a, "auc")) for a in attacks]
            print(
                f"  {s:<26}"
                f" {vals[0]:>9} {vals[1]:>10}"
                f" {vals[2]:>9} {vals[3]:>9} {vals[4]:>9}"
            )
        print()

    # FPR
    print("\n### FPR Control (lower = better)\n")
    h = (
        f"{'Dataset':<28}"
        f" {'QMIA':>8} {'Worst':>8}"
        f" {'LiRA-10':>8} {'LiRA-50':>8} {'LiRA-100':>8}"
    )
    print(h)
    print("\u2500" * len(h))
    for s in sns:
        vals = [_v(_g(results, s, a, "fpr")) for a in attacks]
        print(
            f"  {s:<26}"
            f" {vals[0]:>8} {vals[1]:>8}"
            f" {vals[2]:>8} {vals[3]:>8} {vals[4]:>8}"
        )

    # Speed
    print("\n\n### Speed (seconds)\n")
    h = (
        f"{'Dataset':<28}"
        f" {'QMIA':>9} {'WorstCase':>10}"
        f" {'LiRA-10':>9} {'LiRA-50':>9} {'LiRA-100':>9}"
    )
    print(h)
    print("\u2500" * len(h))
    for s in sns:
        vals = [_vt(_g(results, s, a, "time")) for a in attacks]
        print(
            f"  {s:<26}"
            f" {vals[0]:>9} {vals[1]:>10}"
            f" {vals[2]:>9} {vals[3]:>9} {vals[4]:>9}"
        )

    n_scenarios = len(sns)
    n_runs = len(results)
    print(f"\nTotal: {n_runs} runs across {n_scenarios} scenarios")


if __name__ == "__main__":
    print("Running full QMIA benchmark...\n")
    _print_tables(_run_all())
