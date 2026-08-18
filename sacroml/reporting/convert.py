"""Convert legacy SACRO-ML ``report.json`` files into the new report format.

Older SACRO-ML versions write a flat JSON object keyed by
``"<attack_name>_<log_id>"``.  The new reporting pipeline expects a nested,
catalog-enriched document validated by
``sacroml/reporting/sacroml_attack_report.schema.json``.

The conversion:

1. wraps the legacy experiments under a top-level ``"attacks"`` key;
2. injects the four catalogs (``metric_catalog``, ``parameter_catalog``,
   ``attack_category_catalog``, ``attack_catalog``) from the bundled
   ``catalog_definitions.json``;
3. validates the result against the JSON schema; and
4. warns about any metric, parameter, attack or attack category observed in the
   report that is not present in the catalogs (the conversion still succeeds).

Two small normalisations keep real-world legacy reports schema-valid: a
placeholder ``sacroml_version`` is injected when missing, and an empty
``attack_instance_logger`` is supplied for instance-less attacks (e.g.
structural attacks).  Anything else that does not match the schema -- such as a
non-scalar metric or a stray metadata key -- is surfaced as a schema error
rather than silently rewritten.

Curve-valued arrays (``fpr`` / ``tpr`` / ``roc_thresh``) are passed through
untouched; ``roc_thresh`` legitimately starts with ``null``, which the schema
does not yet permit, so such violations are reported as notices rather than
errors.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

REPORT_SCHEMA_VERSION = "1.2"

_HERE = Path(__file__).parent
SCHEMA_PATH = _HERE / "sacroml_attack_report.schema.json"
CATALOG_DEFINITIONS_PATH = _HERE / "catalog_definitions.json"

# Top-level keys that are *not* experiments in an already-converted report.
_NON_EXPERIMENT_KEYS = frozenset(
    {
        "report_schema_version",
        "metric_catalog",
        "parameter_catalog",
        "attack_category_catalog",
        "attack_catalog",
        "attacks",
    }
)


@dataclass
class ConversionResult:
    """Outcome of converting a legacy report.

    Attributes
    ----------
    report : dict
        The converted report (new format).
    warnings : list[str]
        Non-fatal issues: uncatalogued metrics/parameters/attacks/categories
        and the normalisations that were applied.
    curve_warnings : list[str]
        Schema violations attributable solely to curve-valued arrays
        (``fpr`` / ``tpr`` / ``roc_thresh``); these are expected and benign.
    schema_errors : list[str]
        Genuine schema violations that make the output invalid.
    coverage : dict
        Per-dimension list of uncatalogued names.
    """

    report: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    curve_warnings: list[str] = field(default_factory=list)
    schema_errors: list[str] = field(default_factory=list)
    coverage: dict[str, list[str]] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Return whether the converted report passes schema validation.

        Curve-array violations do not count as failures.
        """
        return not self.schema_errors


def _load_json(path: Path) -> dict[str, Any]:
    """Load a bundled JSON data file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _is_experiment(value: Any) -> bool:
    """Return whether a top-level value looks like a legacy experiment."""
    return isinstance(value, dict) and (
        "attack_experiment_logger" in value or "metadata" in value or "log_id" in value
    )


def _extract_experiments(data: dict[str, Any]) -> dict[str, Any]:
    """Return the experiment mapping, whether the input is flat or wrapped."""
    if isinstance(data.get("attacks"), dict):
        return dict(data["attacks"])  # already wrapped (idempotent path)
    return {
        key: value
        for key, value in data.items()
        if key not in _NON_EXPERIMENT_KEYS and _is_experiment(value)
    }


def _normalise_experiment(
    exp_key: str, experiment: dict[str, Any], warnings: list[str]
) -> dict[str, Any]:
    """Apply the minimal normalisations needed for schema validity."""
    exp = dict(experiment)
    exp.setdefault("log_id", exp_key)
    exp.setdefault("log_time", "")

    metadata = dict(exp.get("metadata") or {})
    if "sacroml_version" not in metadata:
        metadata["sacroml_version"] = "unknown"
        warnings.append(
            f"Experiment '{exp_key}': metadata.sacroml_version was missing; "
            "set to 'unknown'."
        )
    exp["metadata"] = metadata

    # Instance-less attacks (e.g. structural attacks) have no per-instance
    # metrics; supply an empty logger so the result still satisfies the schema.
    logger = exp.get("attack_experiment_logger")
    instances = logger.get("attack_instance_logger") if isinstance(logger, dict) else {}
    if not isinstance(instances, dict):
        instances = {}
    exp["attack_experiment_logger"] = {"attack_instance_logger": instances}
    return exp


def _compile_pattern_metrics(metric_catalog: dict[str, Any]) -> list[re.Pattern]:
    """Compile the regexes declared in ``metric_catalog.pattern_metrics``."""
    patterns: list[re.Pattern] = []
    for entry in metric_catalog.get("pattern_metrics", []):
        pattern = entry.get("pattern") if isinstance(entry, dict) else None
        if pattern:
            patterns.append(re.compile(pattern))
    return patterns


def _uncatalogued(
    seen: set[str], explicit: set[str], patterns: list[re.Pattern] | None = None
) -> list[str]:
    """Return the sorted names in ``seen`` that no catalog entry covers."""
    patterns = patterns or []
    return sorted(
        name
        for name in seen
        if name not in explicit and not any(p.match(name) for p in patterns)
    )


def _compute_coverage(
    report: dict[str, Any], catalogs: dict[str, Any]
) -> tuple[dict[str, list[str]], list[str]]:
    """Diff observed metrics/parameters/attacks/categories vs the catalogs."""
    seen_metrics: set[str] = set()
    seen_params: set[str] = set()
    seen_attacks: set[str] = set()

    for exp in report["attacks"].values():
        instances = exp["attack_experiment_logger"]["attack_instance_logger"]
        for inst in instances.values():
            if isinstance(inst, dict):
                seen_metrics.update(inst.keys())
        metadata = exp.get("metadata", {})
        for key in ("global_metrics", "baseline_global_metrics"):
            values = metadata.get(key, {})
            if isinstance(values, dict):
                seen_metrics.update(values)
        params = metadata.get("attack_params", {})
        if isinstance(params, dict):
            seen_params.update(params)
        seen_attacks.add(metadata.get("attack_name", ""))
    seen_attacks.discard("")

    metric_catalog = catalogs["metric_catalog"]
    catalog_attacks = catalogs["attack_catalog"].get("attacks", {})
    # Categories referenced by the attacks we can resolve in the catalog.
    seen_categories = {
        catalog_attacks[name]["attack_category"]
        for name in seen_attacks
        if name in catalog_attacks and "attack_category" in catalog_attacks[name]
    }

    coverage = {
        "metrics": _uncatalogued(
            seen_metrics,
            set(metric_catalog.get("metrics", {})),
            _compile_pattern_metrics(metric_catalog),
        ),
        "parameters": _uncatalogued(
            seen_params, set(catalogs["parameter_catalog"].get("parameters", {}))
        ),
        "attacks": _uncatalogued(seen_attacks, set(catalog_attacks)),
        "attack_categories": _uncatalogued(
            seen_categories,
            set(catalogs["attack_category_catalog"].get("categories", {})),
        ),
    }

    labels = {
        "metrics": "metric",
        "parameters": "parameter",
        "attacks": "attack",
        "attack_categories": "attack category",
    }
    warnings = [
        f"Uncatalogued {labels[dim]}: '{name}' is not present in the {dim} catalog."
        for dim, names in coverage.items()
        for name in names
    ]
    return coverage, warnings


def _is_curve_violation(error: Any, report: dict[str, Any]) -> bool:
    """Return whether a schema error is caused solely by a curve array.

    Curve-valued metrics (``fpr`` / ``tpr`` / ``roc_thresh``) are stored as
    raw arrays that may contain ``null`` (e.g. ``roc_thresh[0]``).  These are
    a known schema limitation and are reported as warnings, not errors.
    """
    path = list(error.absolute_path)
    # Expected path: attacks / <exp> / attack_experiment_logger /
    #                attack_instance_logger / instance_<n> / <metric> [/ idx]
    if len(path) < 6:
        return False
    if path[0] != "attacks" or path[2] != "attack_experiment_logger":
        return False
    if path[3] != "attack_instance_logger":
        return False
    metric = path[5]
    if not isinstance(metric, str):
        return False
    try:
        value = report["attacks"][path[1]]["attack_experiment_logger"][
            "attack_instance_logger"
        ][path[4]][metric]
    except (KeyError, TypeError, IndexError):
        return False
    return isinstance(value, list)


def _validate(report: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Validate ``report`` against the schema.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(schema_errors, curve_warnings)`` -- human-readable messages.
    """
    validator = Draft7Validator(_load_json(SCHEMA_PATH))
    schema_errors: list[str] = []
    curve_warnings: list[str] = []
    for error in sorted(validator.iter_errors(report), key=str):
        location = "/".join(str(p) for p in error.absolute_path) or "<root>"
        detail = error.message
        if len(detail) > 200:
            detail = detail[:200] + "... (truncated)"
        message = f"{location}: {detail}"
        if _is_curve_violation(error, report):
            curve_warnings.append(message)
        else:
            schema_errors.append(message)
    return schema_errors, curve_warnings


def convert_report(data: dict[str, Any], *, validate: bool = True) -> ConversionResult:
    """Convert an in-memory legacy report dictionary to the new format.

    Parameters
    ----------
    data : dict
        The parsed legacy ``report.json`` (flat) or an already-wrapped report.
    validate : bool, default True
        Whether to validate the converted report against the JSON schema.

    Returns
    -------
    ConversionResult
        The converted report plus warnings, schema errors and a coverage
        summary.
    """
    warnings: list[str] = []
    experiments: dict[str, Any] = {}
    if isinstance(data, dict):
        experiments = _extract_experiments(data)
    else:
        warnings.append(
            f"Top-level report was a {type(data).__name__}, not an object; "
            "no experiments could be extracted."
        )

    catalogs = _load_json(CATALOG_DEFINITIONS_PATH)
    report: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "metric_catalog": catalogs["metric_catalog"],
        "parameter_catalog": catalogs["parameter_catalog"],
        "attack_category_catalog": catalogs["attack_category_catalog"],
        "attack_catalog": catalogs["attack_catalog"],
        "attacks": {
            key: _normalise_experiment(key, exp, warnings)
            for key, exp in experiments.items()
            if isinstance(exp, dict)
        },
    }

    coverage, coverage_warnings = _compute_coverage(report, catalogs)
    warnings.extend(coverage_warnings)

    schema_errors: list[str] = []
    curve_warnings: list[str] = []
    if validate:
        schema_errors, curve_warnings = _validate(report)

    return ConversionResult(
        report=report,
        warnings=warnings,
        curve_warnings=curve_warnings,
        schema_errors=schema_errors,
        coverage=coverage,
    )


def convert_report_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    validate: bool = True,
    indent: int = 2,
) -> ConversionResult:
    """Convert a legacy ``report.json`` file and write the new format to disk.

    Parameters
    ----------
    input_path : str | Path
        Path to the legacy report.
    output_path : str | Path
        Path to write the converted report to.
    validate : bool, default True
        Whether to validate the converted report against the JSON schema.
    indent : int, default 2
        Indentation for the written JSON.

    Returns
    -------
    ConversionResult
        The conversion outcome.
    """
    with open(input_path, encoding="utf-8") as fh:
        data = json.load(fh)

    result = convert_report(data, validate=validate)

    output_path = Path(output_path)
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result.report, fh, indent=indent)
        fh.write("\n")

    return result
