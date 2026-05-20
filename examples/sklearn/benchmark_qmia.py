"""Benchmark QMIA, WorstCase, and LiRA across configurable scenarios.

Compares QMIA against existing MIA attacks across synthetic and sklearn
datasets. Outputs comparison tables to stdout and/or structured JSON for
downstream analysis with ``summarize_qmia_lira_benchmark.py``.

Usage:
    python benchmark_qmia.py                          # stdout, default scenarios
    python benchmark_qmia.py --output json            # JSON only
    python benchmark_qmia.py --output both            # stdout + JSON
    python benchmark_qmia.py --dataset-source sklearn # use sklearn datasets
    python benchmark_qmia.py --scenarios-json scenarios.json
    python benchmark_qmia.py --include-worstcase      # also benchmark WorstCase
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sacroml.attacks.likelihood_attack import LIRAAttack
from sacroml.attacks.qmia_attack import QMIAAttack
from sacroml.attacks.target import Target
from sacroml.attacks.worst_case_attack import WorstCaseAttack

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")


@dataclass
class Scenario:
    """Synthetic benchmark scenario settings."""

    name: str
    n_samples: int
    n_features: int
    class_sep: float
    random_state: int
    n_classes: int = 2


DEFAULT_SCENARIOS: list[Scenario] = [
    Scenario(
        name="small_binary", n_samples=240, n_features=8, class_sep=1.25, random_state=7
    ),
    Scenario(
        name="medium_binary",
        n_samples=600,
        n_features=16,
        class_sep=0.9,
        random_state=13,
    ),
    Scenario(
        name="small_multi",
        n_samples=500,
        n_features=10,
        class_sep=1.5,
        random_state=9,
        n_classes=3,
    ),
]


def _build_target(
    X: Any,
    y: Any,
    name: str,
    *,
    random_state: int,
    rf_estimators: int,
    test_size: float,
) -> Target:
    """Build a Target from feature/label arrays.

    Falls back to a non-stratified split if the data is too small or
    imbalanced for stratify=y to satisfy.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
    except ValueError:
        print(
            f"  Warning: stratified split failed for {name!r}; "
            f"falling back to random split.",
            file=sys.stderr,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=None, random_state=random_state
        )
    model = RandomForestClassifier(
        n_estimators=rf_estimators, random_state=random_state
    )
    model.fit(X_train, y_train)
    target = Target(
        model=model,
        dataset_name=name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_train_orig=X_train,
        y_train_orig=y_train,
        X_test_orig=X_test,
        y_test_orig=y_test,
    )
    for idx in range(X.shape[1]):
        target.add_feature(f"V{idx}", [idx], "float")
    return target


def _build_synthetic_target(
    scenario: Scenario, rf_estimators: int, test_size: float
) -> Target:
    """Build a Target from a synthetic Scenario."""
    X, y = make_classification(
        n_samples=scenario.n_samples,
        n_features=scenario.n_features,
        n_informative=max(4, scenario.n_features // 2),
        n_redundant=0,
        n_repeated=0,
        n_classes=scenario.n_classes,
        n_clusters_per_class=1,
        class_sep=scenario.class_sep,
        random_state=scenario.random_state,
    )
    return _build_target(
        X,
        y,
        scenario.name,
        random_state=scenario.random_state,
        rf_estimators=rf_estimators,
        test_size=test_size,
    )


def _load_sklearn_dataset(name: str) -> tuple[Any, Any, str]:
    """Load a supported sklearn dataset preset."""
    if name == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True, as_frame=False)
        return X, y, "breast_cancer"
    if name == "wine_binary":
        X, y = load_wine(return_X_y=True, as_frame=False)
        y_binary = (y == 0).astype(int)
        return X, y_binary, "wine_binary_class0_vs_rest"
    raise ValueError(
        "Unsupported sklearn dataset preset. Use one of: breast_cancer, wine_binary."
    )


def _load_scenarios(args: argparse.Namespace) -> list[Scenario]:
    """Load scenarios from JSON file or use defaults."""
    if args.scenarios_json is None:
        return DEFAULT_SCENARIOS
    payload = json.loads(Path(args.scenarios_json).read_text(encoding="utf-8"))
    return [Scenario(**item) for item in payload]


def _failed_row(
    scenario_name: str, attack_name: str, elapsed: float, status: str
) -> dict[str, Any]:
    """Build a row representing a failed/skipped attack run."""
    return {
        "scenario": scenario_name,
        "attack": attack_name,
        "seconds": round(elapsed, 6),
        "status": status,
    }


def _run_attack(
    scenario_name: str, attack_name: str, attack: Any, target: Target
) -> dict[str, Any]:
    """Run one attack and return a row of timing + metrics, or a failure row.

    Successful runs return a row with metric fields (``AUC``, ``TPR``, ...).
    Skipped or failed runs return a row with a single ``status`` string
    describing why (no metric fields).
    """
    started = time.perf_counter()
    output = attack.attack(target)
    elapsed = time.perf_counter() - started

    if not output:
        return _failed_row(
            scenario_name, attack_name, elapsed, "not_attackable_or_empty"
        )

    if output.get("status") == "failed":
        return _failed_row(
            scenario_name,
            attack_name,
            elapsed,
            output.get("fail_reason", "attack reported failed status"),
        )

    try:
        metrics = output["attack_experiment_logger"]["attack_instance_logger"][
            "instance_0"
        ]
    except (KeyError, TypeError):
        return _failed_row(
            scenario_name, attack_name, elapsed, "unexpected_output_structure"
        )

    row: dict[str, Any] = {
        "scenario": scenario_name,
        "attack": attack_name,
        "seconds": round(elapsed, 6),
        "AUC": float(metrics["AUC"]),
        "Advantage": float(metrics["Advantage"]),
        "TPR": float(metrics["TPR"]),
        "FPR": float(metrics["FPR"]),
    }
    for k in ("TPR@0.1", "TPR@0.01", "TPR@0.001", "observed_public_fpr"):
        if k in metrics:
            row[k] = float(metrics[k])
    return row


def _fmt_metric(val: float | None) -> str:
    """Format a metric value (missing or NaN values render as "—")."""
    if val is None or (isinstance(val, float) and val != val):
        return "—"
    return f"{val:.3f}"


def _fmt_seconds(val: float | None) -> str:
    """Format a wall-time value in seconds (missing values render as "—")."""
    if val is None:
        return "—"
    return f"{val:.2f}s"


def _lookup(rows: list[dict[str, Any]], scenario: str, attack: str, field_: str) -> Any:
    """Find a row's ``field`` by ``(scenario, attack)``, or ``None``."""
    for r in rows:
        if r["scenario"] == scenario and r["attack"] == attack:
            return r.get(field_)
    return None


def _print_section(
    rows: list[dict[str, Any]],
    scenarios: list[str],
    attacks: list[str],
    title: str,
    field_: str,
    fmt_fn: Any = _fmt_metric,
) -> None:
    """Print one formatted comparison table section."""
    print(f"\n### {title}\n")
    header = f"{'Dataset':<28}" + "".join(f" {a:>11}" for a in attacks)
    print(header)
    print("-" * len(header))
    for s in scenarios:
        line = f"  {s:<26}"
        for a in attacks:
            line += f" {fmt_fn(_lookup(rows, s, a, field_)):>11}"
        print(line)


def _print_tables(rows: list[dict[str, Any]]) -> None:
    """Print AUC, TPR@FPR, FPR-control and runtime comparison tables."""
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    attacks = list(dict.fromkeys(r["attack"] for r in rows))

    _print_section(rows, scenarios, attacks, "AUC Comparison", "AUC")
    for fpr_key, label in [
        ("TPR@0.1", "TPR @ FPR=0.1 (higher = better)"),
        ("TPR@0.01", "TPR @ FPR=0.01 (higher = better)"),
        ("TPR@0.001", "TPR @ FPR=0.001 (higher = better)"),
    ]:
        _print_section(rows, scenarios, attacks, label, fpr_key)
    _print_section(rows, scenarios, attacks, "FPR Control (lower = better)", "FPR")
    _print_section(
        rows, scenarios, attacks, "Speed (seconds)", "seconds", fmt_fn=_fmt_seconds
    )
    print(f"\nTotal: {len(rows)} runs across {len(scenarios)} scenarios")


def _write_outputs(
    out_json: Path,
    out_csv: Path | None,
    args: argparse.Namespace,
    scenarios: list[Scenario],
    rows: list[dict[str, Any]],
) -> None:
    """Write benchmark outputs to JSON (and CSV if requested)."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "dataset_source": args.dataset_source,
            "sklearn_datasets": args.sklearn_datasets,
            "dataset_random_state": args.dataset_random_state,
            "rf_estimators": args.rf_estimators,
            "test_size": args.test_size,
            "qmia_alpha": args.qmia_alpha,
            "qmia_max_iter": args.qmia_max_iter,
            "lira_shadow_models": args.lira_shadow_models,
            "include_worstcase": args.include_worstcase,
        },
        "scenarios": [asdict(s) for s in scenarios],
        "results": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = sorted({key for row in rows for key in row})
        with out_csv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated string into a list of ints."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_name_list(value: str) -> list[str]:
    """Parse a comma-separated string into a list of trimmed names."""
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        choices=["stdout", "json", "both"],
        default="stdout",
        help="Where to send results.",
    )
    parser.add_argument(
        "--dataset-source",
        type=str,
        choices=["synthetic", "sklearn"],
        default="synthetic",
        help="Dataset source.",
    )
    parser.add_argument(
        "--scenarios-json",
        type=str,
        default=None,
        help="Path to a JSON file containing a list of scenario objects.",
    )
    parser.add_argument(
        "--sklearn-datasets",
        type=_parse_name_list,
        default=["breast_cancer", "wine_binary"],
        help="Comma-separated sklearn dataset presets when --dataset-source=sklearn.",
    )
    parser.add_argument(
        "--lira-shadow-models",
        type=_parse_int_list,
        default=[10, 50],
        help='Comma-separated list of LiRA shadow-model counts, e.g. "10,50,100".',
    )
    parser.add_argument(
        "--include-worstcase",
        action="store_true",
        help="Also benchmark WorstCaseAttack.",
    )
    parser.add_argument("--rf-estimators", type=int, default=50)
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--qmia-alpha", type=float, default=0.01)
    parser.add_argument("--qmia-max-iter", type=int, default=100)
    parser.add_argument("--dataset-random-state", type=int, default=7)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument(
        "--out-json",
        type=str,
        default=f"outputs/benchmarks/qmia_{timestamp}.json",
        help="Output JSON path (used when --output is json or both).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional CSV output path.",
    )
    return parser.parse_args()


def _build_cases(
    args: argparse.Namespace, scenarios: list[Scenario]
) -> list[tuple[str, Target, int]]:
    """Build the (name, target, random_state) cases to benchmark."""
    cases: list[tuple[str, Target, int]] = []
    if args.dataset_source == "synthetic":
        for scenario in scenarios:
            target = _build_synthetic_target(
                scenario, args.rf_estimators, args.test_size
            )
            cases.append((scenario.name, target, scenario.random_state))
    else:
        for ds in args.sklearn_datasets:
            X, y, resolved = _load_sklearn_dataset(ds)
            target = _build_target(
                X,
                y,
                resolved,
                random_state=args.dataset_random_state,
                rf_estimators=args.rf_estimators,
                test_size=args.test_size,
            )
            cases.append((resolved, target, args.dataset_random_state))
    return cases


def _benchmark_case(
    args: argparse.Namespace,
    temp_base: Path,
    case_name: str,
    target: Target,
    case_random_state: int,
) -> list[dict[str, Any]]:
    """Run all configured attacks against one case."""
    rows: list[dict[str, Any]] = [
        _run_attack(
            case_name,
            "qmia",
            QMIAAttack(
                output_dir=str(temp_base / f"{case_name}_qmia"),
                write_report=False,
                alpha=args.qmia_alpha,
                max_iter=args.qmia_max_iter,
                random_state=case_random_state,
            ),
            target,
        )
    ]
    if args.include_worstcase:
        rows.append(
            _run_attack(
                case_name,
                "worstcase",
                WorstCaseAttack(
                    output_dir=str(temp_base / f"{case_name}_worstcase"),
                    write_report=False,
                    n_reps=3,
                ),
                target,
            )
        )
    if len(np.unique(target.y_train)) == 2:
        for n_shadow in args.lira_shadow_models:
            rows.append(
                _run_attack(
                    case_name,
                    f"lira_{n_shadow}",
                    LIRAAttack(
                        output_dir=str(temp_base / f"{case_name}_lira_{n_shadow}"),
                        write_report=False,
                        n_shadow_models=n_shadow,
                    ),
                    target,
                )
            )
    return rows


def main() -> None:
    """Run benchmark sweep."""
    args = parse_args()
    scenarios: list[Scenario] = (
        _load_scenarios(args) if args.dataset_source == "synthetic" else []
    )

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="qmia_bench_") as tmpdir:
        temp_base = Path(tmpdir)
        for case_name, target, case_random_state in _build_cases(args, scenarios):
            rows.extend(
                _benchmark_case(args, temp_base, case_name, target, case_random_state)
            )

    if args.output in ("stdout", "both"):
        _print_tables(rows)

    if args.output in ("json", "both"):
        out_json = Path(args.out_json)
        out_csv = Path(args.out_csv) if args.out_csv else None
        _write_outputs(out_json, out_csv, args, scenarios, rows)
        print(f"\nSaved JSON results to: {out_json}")
        if out_csv is not None:
            print(f"Saved CSV results to: {out_csv}")


if __name__ == "__main__":
    main()
