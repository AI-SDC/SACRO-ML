"""Convert legacy SACRO-ML ``report.json`` files into the new report format.

Older SACRO-ML versions write a flat JSON object keyed by
``"<attack_name>_<log_id>"``.  The new reporting pipeline expects a nested,
catalog-enriched document validated by
``sacroml/reporting/sacroml_attack_report.schema.json``.

This module performs that conversion:

1.  Wrap the top-level experiments under an ``"attacks"`` key.
2.  Normalise experiment metadata so it satisfies the schema (injecting a
    placeholder ``sacroml_version`` for very old reports, ensuring an
    ``attack_instance_logger`` exists for instance-less structural attacks,
    coercing non-scalar ``global_metrics`` values, etc.).
3.  Inject the four human-readable catalogs (``metric_catalog``,
    ``parameter_catalog``, ``attack_category_catalog``, ``attack_catalog``)
    from a bundled common-definitions file.
4.  Diff everything observed in the report against the catalogs and emit
    warnings for metrics, parameters, attacks and attack categories that are
    not catalogued (the conversion still succeeds).
5.  Validate the result against the JSON schema.  Schema violations caused
    purely by curve-valued arrays (``fpr`` / ``tpr`` / ``roc_thresh`` -- the
    latter legitimately starts with ``null``) are downgraded to warnings,
    since they are a known limitation tracked separately; any other schema
    violation is reported as an error.

The conversion never mutates curve arrays: they are passed through verbatim.
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

# Metadata keys permitted by the schema (``additionalProperties: false``).
_ALLOWED_METADATA_KEYS = frozenset(
    {
        "sacroml_version",
        "attack_name",
        "attack_params",
        "global_metrics",
        "baseline_global_metrics",
        "target_model",
        "target_model_params",
        "target_train_params",
    }
)
_ATTACK_KEY_RE = re.compile(r"^[A-Za-z0-9 _\-]+$")
_INSTANCE_KEY_RE = re.compile(r"^instance_[0-9]+$")


@dataclass
class ConversionResult:
    """Outcome of converting a legacy report.

    Attributes
    ----------
    report : dict
        The converted report (new format).
    warnings : list[str]
        Non-fatal issues: uncatalogued metrics/parameters/attacks/categories
        and structural normalisations that were applied.
    curve_warnings : list[str]
        Schema violations attributable solely to curve-valued arrays
        (``fpr`` / ``tpr`` / ``roc_thresh``); these are expected and benign.
    schema_errors : list[str]
        Genuine schema violations that make the output invalid.
    coverage : dict
        Per-dimension ``{"covered": [...], "missing": [...]}`` summary.
    """

    report: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    curve_warnings: list[str] = field(default_factory=list)
    schema_errors: list[str] = field(default_factory=list)
    coverage: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Return whether the converted report passes schema validation.

        Curve-array violations do not count as failures.
        """
        return not self.schema_errors

    def summary_dict(self) -> dict[str, Any]:
        """Return a machine-readable summary of the conversion outcome."""
        return {
            "is_valid": self.is_valid,
            "warnings": len(self.warnings),
            "curve_warnings": len(self.curve_warnings),
            "schema_errors": len(self.schema_errors),
            "coverage": {
                dim: {
                    "covered": len(summary["covered"]),
                    "missing": len(summary["missing"]),
                }
                for dim, summary in self.coverage.items()
            },
        }

    def summary(self, *, validated: bool = True) -> str:
        """Return a human-readable, multi-line summary of the conversion.

        Parameters
        ----------
        validated : bool, default True
            Whether schema validation was performed; controls the final
            "schema-valid" / "NOT schema-valid" line.
        """
        lines: list[str] = []
        for dim, summary in self.coverage.items():
            lines.append(
                f"  {dim}: {len(summary['covered'])} catalogued, "
                f"{len(summary['missing'])} uncatalogued"
            )
        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            lines.extend(f"  - {w}" for w in self.warnings)
        if self.curve_warnings:
            lines.append(
                f"\nCurve-array notices ({len(self.curve_warnings)}): "
                "fpr/tpr/roc_thresh arrays are passed through unchanged and "
                "do not strictly validate yet."
            )
            lines.extend(f"  - {w}" for w in self.curve_warnings)
        if self.schema_errors:
            lines.append(f"\nSchema errors ({len(self.schema_errors)}):")
            lines.extend(f"  - {e}" for e in self.schema_errors)
            lines.append("\nConverted report is NOT schema-valid.")
        elif validated:
            lines.append("\nConverted report is schema-valid.")
        return "\n".join(lines)


def _load_catalog_definitions() -> dict[str, Any]:
    """Load the bundled common catalog definitions."""
    with open(CATALOG_DEFINITIONS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _load_schema() -> dict[str, Any]:
    """Load the bundled attack-report JSON schema."""
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _is_experiment(value: Any) -> bool:
    """Return whether a top-level value looks like a legacy experiment."""
    return isinstance(value, dict) and (
        "attack_experiment_logger" in value or "metadata" in value or "log_id" in value
    )


def _extract_experiments(data: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    """Return the experiment mapping, whether the input is flat or wrapped."""
    if isinstance(data.get("attacks"), dict):
        # Already wrapped (idempotent path).
        return dict(data["attacks"])

    experiments: dict[str, Any] = {}
    for key, value in data.items():
        if key in _NON_EXPERIMENT_KEYS:
            continue
        if _is_experiment(value):
            experiments[key] = value
        else:
            warnings.append(
                f"Top-level key '{key}' does not look like an experiment "
                "and was dropped during conversion."
            )
    return experiments


def _coerce_scalar_metrics(
    metrics: Any, where: str, warnings: list[str]
) -> dict[str, Any]:
    """Ensure a (baseline_)global_metrics mapping holds only scalars."""
    if not isinstance(metrics, dict):
        warnings.append(f"{where} was not an object; replaced with empty object.")
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in metrics.items():
        if value is None or isinstance(value, (str, bool, int, float)):
            cleaned[key] = value
        else:
            cleaned[key] = json.dumps(value, default=str)
            warnings.append(
                f"{where}['{key}'] held a non-scalar value; "
                "serialised to a JSON string to satisfy the schema."
            )
    return cleaned


def _normalise_instance_logger(
    logger: Any, exp_key: str, warnings: list[str]
) -> dict[str, Any]:
    """Return an ``attack_instance_logger`` with schema-conforming keys."""
    if not isinstance(logger, dict):
        return {}
    normalised: dict[str, Any] = {}
    next_index = 0
    for key, value in logger.items():
        if _INSTANCE_KEY_RE.match(key):
            normalised[key] = value
            continue
        while (new_key := f"instance_{next_index}") in logger or new_key in normalised:
            next_index += 1
        normalised[new_key] = value
        next_index += 1
        warnings.append(
            f"Experiment '{exp_key}': instance key '{key}' did not match "
            f"'instance_<n>'; renamed to '{new_key}'."
        )
    return normalised


def _normalise_metadata(
    exp_key: str, raw_metadata: Any, warnings: list[str]
) -> dict[str, Any]:
    """Normalise an experiment's metadata so it conforms to the schema."""
    metadata = dict(raw_metadata or {})

    for key in sorted(set(metadata) - _ALLOWED_METADATA_KEYS):
        warnings.append(
            f"Experiment '{exp_key}': metadata key '{key}' is not permitted "
            "by the schema and was dropped."
        )
        metadata.pop(key, None)

    if "sacroml_version" not in metadata:
        metadata["sacroml_version"] = "unknown"
        warnings.append(
            f"Experiment '{exp_key}': metadata.sacroml_version was missing; "
            "set to 'unknown'."
        )
    else:
        metadata["sacroml_version"] = str(metadata["sacroml_version"])

    metadata["attack_name"] = str(
        metadata.get("attack_name", exp_key.rsplit("_", 1)[0])
    )
    if not isinstance(metadata.get("attack_params"), dict):
        metadata["attack_params"] = {}

    metadata["global_metrics"] = _coerce_scalar_metrics(
        metadata.get("global_metrics", {}),
        f"Experiment '{exp_key}': metadata.global_metrics",
        warnings,
    )
    if "baseline_global_metrics" in metadata:
        metadata["baseline_global_metrics"] = _coerce_scalar_metrics(
            metadata["baseline_global_metrics"],
            f"Experiment '{exp_key}': metadata.baseline_global_metrics",
            warnings,
        )

    if "target_model" in metadata and not isinstance(metadata["target_model"], str):
        metadata["target_model"] = str(metadata["target_model"])
    for opt in ("target_model_params", "target_train_params"):
        if opt in metadata and not isinstance(metadata[opt], dict):
            metadata[opt] = {}
    return metadata


def _normalise_attack_experiment_logger(
    exp_key: str, raw_logger: Any, warnings: list[str]
) -> dict[str, Any]:
    """Return an ``attack_experiment_logger`` holding only the instance log."""
    ael = raw_logger if isinstance(raw_logger, dict) else {}
    instance_logger = _normalise_instance_logger(
        ael.get("attack_instance_logger", {}), exp_key, warnings
    )
    for key in sorted(set(ael) - {"attack_instance_logger"}):
        warnings.append(
            f"Experiment '{exp_key}': attack_experiment_logger.{key} is not "
            "permitted by the schema and was dropped."
        )
    return {"attack_instance_logger": instance_logger}


def _normalise_experiment(
    exp_key: str, experiment: Any, warnings: list[str]
) -> dict[str, Any]:
    """Normalise a single experiment so it conforms to the schema."""
    if not isinstance(experiment, dict):
        warnings.append(f"Experiment '{exp_key}' was not an object and was skipped.")
        return {}

    exp = dict(experiment)
    exp["log_id"] = str(exp.get("log_id", exp_key))
    if "log_time" not in exp or not isinstance(exp["log_time"], str):
        exp["log_time"] = str(exp.get("log_time", "unknown"))
    exp["metadata"] = _normalise_metadata(exp_key, exp.get("metadata", {}), warnings)
    exp["attack_experiment_logger"] = _normalise_attack_experiment_logger(
        exp_key, exp.get("attack_experiment_logger"), warnings
    )
    return exp


def _compile_pattern_metrics(metric_catalog: dict[str, Any]) -> list[re.Pattern]:
    """Compile the regexes declared in ``metric_catalog.pattern_metrics``."""
    patterns: list[re.Pattern] = []
    for entry in metric_catalog.get("pattern_metrics", []):
        pattern = entry.get("pattern") if isinstance(entry, dict) else None
        if pattern:
            patterns.append(re.compile(pattern))
    return patterns


def _diff(
    seen: set[str], explicit: set[str], patterns: list[re.Pattern] | None = None
) -> tuple[list[str], list[str]]:
    """Split ``seen`` names into catalogued (covered) and missing."""
    patterns = patterns or []
    covered: list[str] = []
    missing: list[str] = []
    for name in sorted(seen):
        if name in explicit or any(p.match(name) for p in patterns):
            covered.append(name)
        else:
            missing.append(name)
    return covered, missing


def _compute_coverage(
    report: dict[str, Any], catalogs: dict[str, Any]
) -> tuple[dict[str, dict[str, list[str]]], list[str]]:
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
        seen_metrics.update(metadata.get("global_metrics", {}).keys())
        seen_metrics.update(metadata.get("baseline_global_metrics", {}).keys())
        seen_params.update(metadata.get("attack_params", {}).keys())
        seen_attacks.add(metadata.get("attack_name", ""))

    seen_attacks.discard("")

    metric_catalog = catalogs["metric_catalog"]
    explicit_metrics = set(metric_catalog.get("metrics", {}).keys())
    pattern_regexes = _compile_pattern_metrics(metric_catalog)
    m_covered, m_missing = _diff(seen_metrics, explicit_metrics, pattern_regexes)

    explicit_params = set(catalogs["parameter_catalog"].get("parameters", {}).keys())
    p_covered, p_missing = _diff(seen_params, explicit_params)

    catalog_attacks = catalogs["attack_catalog"].get("attacks", {})
    a_covered, a_missing = _diff(seen_attacks, set(catalog_attacks.keys()))

    # Categories referenced by the attacks we *can* resolve in the catalog.
    known_categories = set(
        catalogs["attack_category_catalog"].get("categories", {}).keys()
    )
    seen_categories = {
        catalog_attacks[name]["attack_category"]
        for name in seen_attacks
        if name in catalog_attacks and "attack_category" in catalog_attacks[name]
    }
    c_covered, c_missing = _diff(seen_categories, known_categories)

    coverage = {
        "metrics": {"covered": m_covered, "missing": m_missing},
        "parameters": {"covered": p_covered, "missing": p_missing},
        "attacks": {"covered": a_covered, "missing": a_missing},
        "attack_categories": {"covered": c_covered, "missing": c_missing},
    }

    warnings: list[str] = []
    labels = {
        "metrics": "metric",
        "parameters": "parameter",
        "attacks": "attack",
        "attack_categories": "attack category",
    }
    for dim, label in labels.items():
        for name in coverage[dim]["missing"]:
            warnings.append(
                f"Uncatalogued {label}: '{name}' is not present in the {dim} catalog."
            )
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


def _validate(
    report: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Validate ``report`` against the schema.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(schema_errors, curve_warnings)`` -- human-readable messages.
    """
    validator = Draft7Validator(_load_schema())
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
    experiments = _extract_experiments(data, warnings)

    converted_attacks: dict[str, Any] = {}
    for exp_key, experiment in experiments.items():
        safe_key = exp_key
        if not _ATTACK_KEY_RE.match(exp_key):
            safe_key = re.sub(r"[^A-Za-z0-9 _\-]", "_", exp_key)
            warnings.append(
                f"Experiment key '{exp_key}' contained characters not allowed "
                f"by the schema; renamed to '{safe_key}'."
            )
        normalised = _normalise_experiment(exp_key, experiment, warnings)
        if normalised:
            converted_attacks[safe_key] = normalised

    catalogs = _load_catalog_definitions()
    report: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "metric_catalog": catalogs["metric_catalog"],
        "parameter_catalog": catalogs["parameter_catalog"],
        "attack_category_catalog": catalogs["attack_category_catalog"],
        "attack_catalog": catalogs["attack_catalog"],
        "attacks": converted_attacks,
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
