"""Summarize QMIA-vs-LiRA benchmark JSON outputs.

Reports per-scenario winners for:
- fastest runtime
- best AUC
- best AUC-per-second
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    """Load benchmark result rows from a benchmark JSON file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark JSON not found: {path}. Run the benchmark first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "results" not in payload:
        raise ValueError(
            "Expected a benchmark JSON payload with a top-level 'results'."
        )
    return payload["results"]


def _group_by_scenario(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group rows by their ``scenario`` field, preserving insertion order."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        scenario = row.get("scenario", "unknown")
        grouped.setdefault(scenario, []).append(row)
    return grouped


def _safe_auc_per_sec(row: dict[str, Any]) -> float:
    """Return AUC divided by seconds, or ``-inf`` for non-positive seconds."""
    seconds = float(row.get("seconds", 0.0))
    auc = float(row.get("AUC", 0.0))
    return auc / seconds if seconds > 0 else float("-inf")


def _pick_fastest(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the row with the smallest ``seconds``, or ``None`` if none eligible."""
    eligible = [r for r in rows if "seconds" in r]
    if not eligible:
        return None
    return min(eligible, key=lambda r: float(r["seconds"]))


def _pick_best_auc(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the row with the largest ``AUC`` value, or ``None`` if none eligible."""
    eligible = [r for r in rows if "AUC" in r]
    if not eligible:
        return None
    return max(eligible, key=lambda r: float(r["AUC"]))


def _pick_best_auc_per_sec(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the row with the best AUC-per-second, or ``None`` if none eligible."""
    eligible = [r for r in rows if "AUC" in r and "seconds" in r]
    if not eligible:
        return None
    return max(eligible, key=_safe_auc_per_sec)


def _format_row(row: dict[str, Any]) -> str:
    """Format a single result row for one-line summary display."""
    attack = row.get("attack", "unknown")
    seconds = float(row.get("seconds", float("nan")))
    auc = float(row.get("AUC", float("nan")))
    advantage = float(row.get("Advantage", float("nan")))
    return (
        f"{attack} | secs={seconds:.4f} | AUC={auc:.4f} | "
        f"Advantage={advantage:.4f} | AUC/sec={_safe_auc_per_sec(row):.4f}"
    )


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Print a leaderboard of rows sorted by descending AUC."""
    headers = ("attack", "secs", "AUC", "Adv", "TPR", "FPR", "AUC/sec")
    attack_width = max(
        len(headers[0]),
        *(len(str(r.get("attack", "unknown"))) for r in rows),
    )
    print(
        f"  {'#':>2}  {headers[0]:<{attack_width}}  {headers[1]:>8}  {headers[2]:>8}  "
        f"{headers[3]:>8}  {headers[4]:>8}  {headers[5]:>8}  {headers[6]:>10}"
    )
    sep = f"  {'-' * 2}  {'-' * attack_width}"
    sep += f"  {'-' * 8}" * 5 + f"  {'-' * 10}"
    print(sep)
    sorted_rows = sorted(rows, key=lambda r: float(r.get("AUC", 0.0)), reverse=True)
    for idx, row in enumerate(sorted_rows, start=1):
        attack = str(row.get("attack", "unknown"))
        seconds = float(row.get("seconds", float("nan")))
        auc = float(row.get("AUC", float("nan")))
        advantage = float(row.get("Advantage", float("nan")))
        tpr = float(row.get("TPR", float("nan")))
        fpr = float(row.get("FPR", float("nan")))
        auc_per_sec = _safe_auc_per_sec(row)
        print(
            f"  {idx:>2}  {attack:<{attack_width}}  {seconds:>8.4f}  {auc:>8.4f}  "
            f"{advantage:>8.4f}  {tpr:>8.4f}  {fpr:>8.4f}  {auc_per_sec:>10.4f}"
        )


def _print_scenario_summary(scenario: str, scenario_rows: list[dict[str, Any]]) -> None:
    """Print per-scenario winners (fastest, best AUC, best AUC/sec) and leaderboard."""
    print(f"\nScenario: {scenario} (runs: {len(scenario_rows)})")
    fastest = _pick_fastest(scenario_rows)
    best_auc = _pick_best_auc(scenario_rows)
    best_auc_per_sec = _pick_best_auc_per_sec(scenario_rows)
    none_msg = "no successful runs"
    fastest_str = _format_row(fastest) if fastest else none_msg
    best_auc_str = _format_row(best_auc) if best_auc else none_msg
    best_per_sec_str = _format_row(best_auc_per_sec) if best_auc_per_sec else none_msg
    print(f"  Fastest:         {fastest_str}")
    print(f"  Best AUC:        {best_auc_str}")
    print(f"  Best AUC / sec:  {best_per_sec_str}")
    print("  Leaderboard (sorted by AUC):")
    _print_table(scenario_rows)


def summarize(path: Path, title: str | None = None) -> None:
    """Summarize a single benchmark JSON file."""
    rows = _load_rows(path)
    grouped = _group_by_scenario(rows)

    if title is None:
        print(f"Summary for: {path}")
    else:
        print(title)
        print(f"Source: {path}")
    print(f"Total runs: {len(rows)} | Scenarios: {len(grouped)}")
    for scenario, scenario_rows in grouped.items():
        _print_scenario_summary(scenario, scenario_rows)


def summarize_multiple(paths: list[Path]) -> None:
    """Summarize multiple benchmark JSON files with a combined view."""
    for idx, path in enumerate(paths, start=1):
        summarize(path, title=f"Summary {idx}/{len(paths)}")
        if idx < len(paths):
            print(f"\n{'=' * 96}\n")

    combined_rows: list[dict[str, Any]] = []
    for path in paths:
        combined_rows.extend(_load_rows(path))
    combined_path_label = ", ".join(str(path) for path in paths)
    grouped = _group_by_scenario(combined_rows)
    print(f"\n{'#' * 96}")
    print("Combined summary (all benchmark files)")
    print(f"Sources: {combined_path_label}")
    print(f"Total runs: {len(combined_rows)} | Scenarios: {len(grouped)}")
    for scenario, scenario_rows in grouped.items():
        _print_scenario_summary(scenario, scenario_rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmark_json",
        type=str,
        nargs="+",
        help="One or more JSON files generated by benchmark_qmia_vs_lira.py.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the summarize script."""
    args = parse_args()
    try:
        paths = [Path(path) for path in args.benchmark_json]
        if len(paths) == 1:
            summarize(paths[0])
        else:
            summarize_multiple(paths)
    except FileNotFoundError as error:
        print(error)


if __name__ == "__main__":
    main()
