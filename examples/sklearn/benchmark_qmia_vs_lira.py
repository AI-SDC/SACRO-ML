"""Reproducible QMIA-vs-LiRA benchmark runner.

This script benchmarks:
- QMIA (Gaussian uncertainty mode)
- QMIA (direct quantile mode)
- LiRA with one or more shadow-model counts

It uses synthetic binary tabular datasets by default, and can also benchmark
against sklearn dataset presets (for development-stage validation).
Results are written to JSON (and optionally CSV).
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sacroml.attacks.likelihood_attack import LIRAAttack
from sacroml.attacks.qmia_attack import QMIAAttack
from sacroml.attacks.target import Target


@dataclass
class Scenario:
    """Synthetic benchmark scenario settings."""

    name: str
    n_samples: int
    n_features: int
    class_sep: float
    random_state: int


DEFAULT_SCENARIOS = [
    Scenario(
        name="small_easy",
        n_samples=240,
        n_features=8,
        class_sep=1.25,
        random_state=7,
    ),
    Scenario(
        name="medium_harder",
        n_samples=600,
        n_features=16,
        class_sep=0.9,
        random_state=13,
    ),
]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _build_target_from_arrays(
    *,
    dataset_name: str,
    X: Any,
    y: Any,
    random_state: int,
    rf_estimators: int,
    test_size: float,
) -> Target:
    """Construct a Target object from feature/label arrays."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    model = RandomForestClassifier(n_estimators=rf_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    target = Target(
        model=model,
        dataset_name=dataset_name,
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


def _build_target_from_scenario(
    scenario: Scenario,
    rf_estimators: int,
    test_size: float,
) -> Target:
    """Construct a Target object for one synthetic scenario."""
    X, y = make_classification(
        n_samples=scenario.n_samples,
        n_features=scenario.n_features,
        n_informative=max(4, scenario.n_features // 2),
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        class_sep=scenario.class_sep,
        random_state=scenario.random_state,
    )
    return _build_target_from_arrays(
        dataset_name=scenario.name,
        X=X,
        y=y,
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
        # QMIA v1 is binary-only, so we stage wine as one-vs-rest.
        y_binary = (y == 0).astype(int)
        return X, y_binary, "wine_binary_class0_vs_rest"
    raise ValueError(
        "Unsupported sklearn dataset preset. Use one of: "
        "breast_cancer,wine_binary"
    )


def _parse_name_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _benchmark_attack(
    scenario_name: str,
    attack_name: str,
    attack: Any,
    target: Target,
) -> dict[str, Any]:
    """Run one attack and return timing + key metrics."""
    started = time.perf_counter()
    output = attack.attack(target)
    elapsed = time.perf_counter() - started

    if not output:
        return {
            "scenario": scenario_name,
            "attack": attack_name,
            "seconds": round(elapsed, 6),
            "status": "not_attackable_or_empty",
        }

    metrics = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]
    row = {
        "scenario": scenario_name,
        "attack": attack_name,
        "seconds": round(elapsed, 6),
        "AUC": float(metrics["AUC"]),
        "Advantage": float(metrics["Advantage"]),
        "TPR": float(metrics["TPR"]),
        "FPR": float(metrics["FPR"]),
    }
    if "observed_public_fpr" in metrics:
        row["observed_public_fpr"] = float(metrics["observed_public_fpr"])
    return row


def _load_scenarios(args: argparse.Namespace) -> list[Scenario]:
    """Load scenarios from JSON file or use defaults."""
    if args.scenarios_json is None:
        return DEFAULT_SCENARIOS

    payload = json.loads(Path(args.scenarios_json).read_text(encoding="utf-8"))
    return [Scenario(**item) for item in payload]


def _write_outputs(
    out_json: Path,
    out_csv: Path | None,
    args: argparse.Namespace,
    scenarios: list[Scenario],
    results: list[dict[str, Any]],
) -> None:
    """Write benchmark outputs to disk."""
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
            "qmia_iterations": args.qmia_iterations,
            "qmia_depth": args.qmia_depth,
            "qmia_learning_rate": args.qmia_learning_rate,
            "qmia_l2_leaf_reg": args.qmia_l2_leaf_reg,
            "qmia_subsample": args.qmia_subsample,
            "qmia_catboost_params_json": args.qmia_catboost_params_json,
            "lira_shadow_models": args.lira_shadow_models,
        },
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = sorted({key for row in results for key in row.keys()})
        with out_csv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)


def _v(val: float | None) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return "\u2014"
    return f"{val:.3f}"


def _vt(val: float | None) -> str:
    if val is None:
        return "\u2014"
    return f"{val:.2f}s"


def _lookup(rows, scen, atk, field):
    for r in rows:
        if r["scenario"] == scen and r["attack"] == atk:
            return r.get(field)
    return None


def _print_table(title, rows, scenarios, attacks, field, fmt):
    """Print one formatted comparison table."""
    print(f"\n{title}\n")
    hdr = f"{'Dataset':<24}"
    for a in attacks:
        hdr += f" {a:>14}"
    print(hdr)
    print("\u2500" * len(hdr))
    for s in scenarios:
        line = f"  {s:<22}"
        for a in attacks:
            val = _lookup(rows, s, a, field)
            line += f" {fmt(val):>14}"
        print(line)


def _print_summary(rows: list[dict[str, Any]]) -> None:
    """Print formatted benchmark tables."""
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    attacks = list(dict.fromkeys(r["attack"] for r in rows))

    _print_table("### AUC Comparison", rows, scenarios, attacks, "AUC", _v)
    _print_table(
        "\n### FPR Control (lower = better)",
        rows, scenarios, attacks, "FPR", _v,
    )
    _print_table(
        "\n### Speed (seconds)",
        rows, scenarios, attacks, "seconds", _vt,
    )
    print(f"\nTotal: {len(rows)} runs across {len(scenarios)} scenarios")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-source",
        type=str,
        choices=["synthetic", "sklearn"],
        default="synthetic",
        help=(
            "Dataset source. "
            "'synthetic' uses make_classification scenarios. "
            "'sklearn' uses built-in sklearn dataset presets."
        ),
    )
    parser.add_argument(
        "--scenarios-json",
        type=str,
        default=None,
        help=(
            "Path to a JSON file containing a list of scenario objects with keys: "
            "name, n_samples, n_features, class_sep, random_state."
        ),
    )
    parser.add_argument(
        "--sklearn-datasets",
        type=_parse_name_list,
        default=["breast_cancer", "wine_binary"],
        help=(
            "Comma-separated sklearn dataset presets used when "
            "--dataset-source=sklearn. Supported: breast_cancer,wine_binary."
        ),
    )
    parser.add_argument(
        "--dataset-random-state",
        type=int,
        default=7,
        help="Random state used for sklearn preset train/test splitting.",
    )
    parser.add_argument(
        "--lira-shadow-models",
        type=_parse_int_list,
        default=[20, 40],
        help='Comma-separated list, e.g. "20,40,100".',
    )
    parser.add_argument("--rf-estimators", type=int, default=50)
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--qmia-alpha", type=float, default=0.01)
    parser.add_argument("--qmia-iterations", type=int, default=20)
    parser.add_argument("--qmia-depth", type=int, default=3)
    parser.add_argument("--qmia-learning-rate", type=float, default=0.05)
    parser.add_argument("--qmia-l2-leaf-reg", type=float, default=3.0)
    parser.add_argument(
        "--qmia-subsample",
        type=float,
        default=0.8,
        help="CatBoost subsample used for stronger tuning sweeps.",
    )
    parser.add_argument(
        "--qmia-catboost-params-json",
        type=str,
        default=None,
        help=(
            "Optional JSON object merged into CatBoost params for QMIA runs. "
            "Example: '{\"min_data_in_leaf\":20,\"bagging_temperature\":1.0}'"
        ),
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=f"outputs/benchmarks/qmia_vs_lira_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    """Run benchmark sweep."""
    args = parse_args()
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv) if args.out_csv else None
    scenarios: list[Scenario] = _load_scenarios(args) if args.dataset_source == "synthetic" else []

    qmia_params = {
        "iterations": args.qmia_iterations,
        "depth": args.qmia_depth,
        "learning_rate": args.qmia_learning_rate,
        "l2_leaf_reg": args.qmia_l2_leaf_reg,
        "subsample": args.qmia_subsample,
    }
    if args.qmia_catboost_params_json is not None:
        qmia_params.update(json.loads(args.qmia_catboost_params_json))

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="qmia_lira_bench_") as tmpdir:
        temp_base = Path(tmpdir)
        benchmark_cases: list[tuple[str, Target, int]] = []
        if args.dataset_source == "synthetic":
            for scenario in scenarios:
                target = _build_target_from_scenario(
                    scenario, args.rf_estimators, args.test_size
                )
                benchmark_cases.append((scenario.name, target, scenario.random_state))
        else:
            for dataset_name in args.sklearn_datasets:
                X, y, resolved_name = _load_sklearn_dataset(dataset_name)
                target = _build_target_from_arrays(
                    dataset_name=resolved_name,
                    X=X,
                    y=y,
                    random_state=args.dataset_random_state,
                    rf_estimators=args.rf_estimators,
                    test_size=args.test_size,
                )
                benchmark_cases.append((resolved_name, target, args.dataset_random_state))

        for case_name, target, case_random_state in benchmark_cases:

            rows.append(
                _benchmark_attack(
                    case_name,
                    "qmia_gaussian",
                    QMIAAttack(
                        output_dir=str(temp_base / f"{case_name}_qmia_gaussian"),
                        write_report=False,
                        alpha=args.qmia_alpha,
                        use_gaussian=True,
                        catboost_params=qmia_params,
                        random_state=case_random_state,
                    ),
                    target,
                )
            )
            rows.append(
                _benchmark_attack(
                    case_name,
                    "qmia_quantile",
                    QMIAAttack(
                        output_dir=str(temp_base / f"{case_name}_qmia_quantile"),
                        write_report=False,
                        alpha=args.qmia_alpha,
                        use_gaussian=False,
                        catboost_params=qmia_params,
                        random_state=case_random_state,
                    ),
                    target,
                )
            )
            for n_shadow in args.lira_shadow_models:
                rows.append(
                    _benchmark_attack(
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

    _write_outputs(out_json, out_csv, args, scenarios, rows)
    _print_summary(rows)
    print(f"\nSaved JSON results to: {out_json}")
    if out_csv is not None:
        print(f"Saved CSV results to: {out_csv}")


if __name__ == "__main__":
    main()
