"""Tests for legacy report.json -> new format conversion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sacroml.main import main
from sacroml.reporting import ConversionResult, convert_report, convert_report_file
from sacroml.reporting.convert import _is_curve_violation

FIXTURES = Path(__file__).parent / "fixtures"
DOCS_EXAMPLES = Path(__file__).parents[2] / "docs" / "source" / "attacks"


def _load(name: str) -> dict:
    """Load a fixture JSON file by name."""
    with open(FIXTURES / name, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def basic_result() -> ConversionResult:
    """Conversion result for the well-formed legacy fixture."""
    return convert_report(_load("legacy_basic.json"))


@pytest.fixture
def edge_result() -> ConversionResult:
    """Conversion result for the edge-case legacy fixture."""
    return convert_report(_load("legacy_edge.json"))


def test_basic_conversion_structure(basic_result: ConversionResult) -> None:
    """A clean legacy report wraps under 'attacks' with all catalogs."""
    report = basic_result.report
    assert report["report_schema_version"]
    for catalog in (
        "metric_catalog",
        "parameter_catalog",
        "attack_category_catalog",
        "attack_catalog",
        "attacks",
    ):
        assert catalog in report
    assert set(report["attacks"]) == {
        "WorstCase attack_de2c5fc0-fb0c-4925-ac4d-26662fe7f786",
        "LiRA Attack_4e9b5db3-f93f-43dc-9a67-1035a68b892a",
    }


def test_basic_conversion_is_schema_valid(
    basic_result: ConversionResult,
) -> None:
    """A clean legacy report converts to a schema-valid document."""
    assert basic_result.is_valid
    assert basic_result.schema_errors == []
    assert basic_result.warnings == []
    assert basic_result.curve_warnings == []


def test_basic_coverage_all_catalogued(
    basic_result: ConversionResult,
) -> None:
    """All metrics/params/attacks in the clean fixture are catalogued."""
    for dim in basic_result.coverage.values():
        assert dim["missing"] == []


def test_conversion_is_idempotent(basic_result: ConversionResult) -> None:
    """Converting an already-converted report is a no-op for the payload."""
    again = convert_report(basic_result.report)
    assert again.report["attacks"] == basic_result.report["attacks"]
    assert again.is_valid


def test_missing_sacroml_version_injected(
    edge_result: ConversionResult,
) -> None:
    """Old reports without sacroml_version get an 'unknown' placeholder."""
    lira = edge_result.report["attacks"]["LiRA Attack_0d8cf41e"]
    assert lira["metadata"]["sacroml_version"] == "unknown"
    assert any("sacroml_version was missing" in w for w in edge_result.warnings)


def test_stray_metadata_key_dropped(edge_result: ConversionResult) -> None:
    """Metadata keys not permitted by the schema are dropped with a warning."""
    lira_meta = edge_result.report["attacks"]["LiRA Attack_0d8cf41e"]["metadata"]
    assert "stray_metadata_key" not in lira_meta
    assert any("stray_metadata_key" in w for w in edge_result.warnings)


def test_non_scalar_global_metric_serialised(
    edge_result: ConversionResult,
) -> None:
    """Non-scalar global_metrics values are serialised to keep schema valid."""
    gm = edge_result.report["attacks"]["LiRA Attack_0d8cf41e"]["metadata"][
        "global_metrics"
    ]
    assert isinstance(gm["nested_metric"], str)
    assert json.loads(gm["nested_metric"]) == {"a": 1, "b": 2}


def test_extra_experiment_logger_sibling_dropped(
    edge_result: ConversionResult,
) -> None:
    """Attack_experiment_logger keeps only attack_instance_logger."""
    ael = edge_result.report["attacks"]["LiRA Attack_0d8cf41e"][
        "attack_experiment_logger"
    ]
    assert set(ael) == {"attack_instance_logger"}
    assert any("dummy_attack_experiment_logger" in w for w in edge_result.warnings)


def test_non_conforming_instance_key_renamed(
    edge_result: ConversionResult,
) -> None:
    """Instance keys not matching 'instance_<n>' are renamed."""
    logger = edge_result.report["attacks"]["LiRA Attack_0d8cf41e"][
        "attack_experiment_logger"
    ]["attack_instance_logger"]
    assert all(k.startswith("instance_") for k in logger)
    assert any("renamed to 'instance_" in w for w in edge_result.warnings)


def test_structural_attack_gets_empty_logger(
    edge_result: ConversionResult,
) -> None:
    """An instance-less structural attack gets an empty instance logger."""
    struct = edge_result.report["attacks"]["Structural Attack_aaaa"]
    assert struct["attack_experiment_logger"]["attack_instance_logger"] == {}


def test_uncatalogued_entries_warn(edge_result: ConversionResult) -> None:
    """Unknown metrics/params/attacks are reported but not fatal."""
    cov = edge_result.coverage
    assert "weird_metric" in cov["metrics"]["missing"]
    assert "weird_param" in cov["parameters"]["missing"]
    assert "Mystery attack" in cov["attacks"]["missing"]
    assert any("Uncatalogued metric" in w for w in edge_result.warnings)
    assert any("Uncatalogued parameter" in w for w in edge_result.warnings)
    assert any("Uncatalogued attack" in w for w in edge_result.warnings)


def test_curve_array_is_warning_not_error(
    edge_result: ConversionResult,
) -> None:
    """Roc_thresh starting with null is a curve warning, not a schema error."""
    assert edge_result.curve_warnings
    assert all("roc_thresh" in w for w in edge_result.curve_warnings)
    assert edge_result.is_valid


def test_non_experiment_top_level_dropped() -> None:
    """Top-level keys that are not experiments are dropped with a warning."""
    result = convert_report({"junk": 123, **_load("legacy_basic.json")})
    assert "junk" not in result.report["attacks"]
    assert any("does not look like an experiment" in w for w in result.warnings)


@pytest.mark.parametrize(
    "example",
    ["report_example_lira.json", "report_example_worstcase.json"],
)
def test_real_docs_examples_validate(example: str) -> None:
    """The real example reports shipped in docs convert and validate."""
    with open(DOCS_EXAMPLES / example, encoding="utf-8") as fh:
        data = json.load(fh)
    result = convert_report(data)
    assert result.is_valid, result.schema_errors


def test_no_validate_skips_validation() -> None:
    """Validate=False produces no schema errors or curve warnings."""
    result = convert_report(_load("legacy_edge.json"), validate=False)
    assert result.schema_errors == []
    assert result.curve_warnings == []
    # Coverage warnings are independent of schema validation.
    assert result.coverage["metrics"]["missing"]


def test_convert_report_file_roundtrip(tmp_path: Path) -> None:
    """Convert_report_file writes valid JSON and returns a result."""
    out = tmp_path / "nested" / "converted.json"
    result = convert_report_file(FIXTURES / "legacy_basic.json", out)
    assert out.is_file()
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["attacks"] == result.report["attacks"]
    assert result.is_valid


def test_cli_convert_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The `sacroml convert-report` CLI converts and exits cleanly."""
    out = tmp_path / "converted.json"
    monkeypatch.setattr(
        "sys.argv",
        ["sacroml", "convert-report", str(FIXTURES / "legacy_basic.json"), str(out)],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert out.is_file()
    assert "schema-valid" in capsys.readouterr().out


def test_defensive_normalisation_paths() -> None:
    """Malformed legacy inputs hit every defensive normalisation branch."""
    legacy = {
        "report_schema_version": "0.9",
        "WorstCase attack/with!chars_xyz": {
            "log_id": 12345,
            "log_time": 17181920,
            "metadata": {
                "attack_name": "WorstCase attack",
                "attack_params": "not a dict",
                "global_metrics": "not a dict",
                "target_model": 42,
                "target_model_params": "not a dict",
                "target_train_params": "not a dict",
            },
            "attack_experiment_logger": "not a dict",
        },
    }
    result = convert_report(legacy)
    assert "WorstCase attack_with_chars_xyz" in result.report["attacks"]
    converted = result.report["attacks"]["WorstCase attack_with_chars_xyz"]
    meta = converted["metadata"]
    assert meta["attack_params"] == {}
    assert meta["global_metrics"] == {}
    assert meta["target_model"] == "42"
    assert meta["target_model_params"] == {}
    assert meta["target_train_params"] == {}
    assert isinstance(converted["log_time"], str)
    assert converted["attack_experiment_logger"]["attack_instance_logger"] == {}
    assert any("renamed to" in w for w in result.warnings)


def test_non_dict_instance_logger_becomes_empty() -> None:
    """A non-dict ``attack_instance_logger`` is replaced with an empty one."""
    legacy = {
        "WorstCase attack_qq": {
            "log_id": "qq",
            "log_time": "now",
            "metadata": {
                "sacroml_version": "1.0",
                "attack_name": "WorstCase attack",
                "attack_params": {},
                "global_metrics": {"AUC": 0.5},
            },
            "attack_experiment_logger": {"attack_instance_logger": "not a dict"},
        }
    }
    result = convert_report(legacy)
    logger = result.report["attacks"]["WorstCase attack_qq"][
        "attack_experiment_logger"
    ]["attack_instance_logger"]
    assert logger == {}


def test_instance_key_collision_during_renaming() -> None:
    """Renaming a non-conforming key bumps past an existing matching key."""
    legacy = {
        "LiRA Attack_zz": {
            "log_id": "zz",
            "log_time": "now",
            "metadata": {
                "sacroml_version": "1.0",
                "attack_name": "LiRA Attack",
                "attack_params": {},
                "global_metrics": {"AUC": 0.5},
            },
            "attack_experiment_logger": {
                "attack_instance_logger": {
                    "weird": {"AUC": 0.1},
                    "instance_0": {"AUC": 0.2},
                }
            },
        }
    }
    result = convert_report(legacy)
    logger = result.report["attacks"]["LiRA Attack_zz"]["attack_experiment_logger"][
        "attack_instance_logger"
    ]
    assert set(logger) == {"instance_0", "instance_1"}


def test_already_wrapped_with_non_dict_experiment() -> None:
    """An already-wrapped report whose experiment isn't a dict is skipped."""
    result = convert_report(
        {"attacks": {"BogusAttack_xx": "not a dict"}}, validate=False
    )
    assert result.report["attacks"] == {}
    assert any("was not an object" in w for w in result.warnings)


def test_non_curve_schema_error_is_reported() -> None:
    """An instance metric with an object value yields a real schema error."""
    legacy = {
        "LiRA Attack_yy": {
            "log_id": "yy",
            "log_time": "now",
            "metadata": {
                "sacroml_version": "1.0",
                "attack_name": "LiRA Attack",
                "attack_params": {},
                "global_metrics": {"AUC": 0.5},
            },
            "attack_experiment_logger": {
                "attack_instance_logger": {
                    "instance_0": {"AUC": {"unexpected": "object"}}
                }
            },
        }
    }
    result = convert_report(legacy)
    assert not result.is_valid
    assert result.schema_errors


class _FakeError:
    """Minimal stand-in for ``jsonschema.exceptions.ValidationError``."""

    def __init__(self, path: list) -> None:
        self.absolute_path = path


@pytest.mark.parametrize(
    "path",
    [
        ["attacks"],  # path < 6
        ["other", "exp", "attack_experiment_logger", "ail", "i0", "AUC"],
        ["attacks", "exp", "wrong", "ail", "i0", "AUC"],
        ["attacks", "exp", "attack_experiment_logger", "wrong", "i0", "AUC"],
        [
            "attacks",
            "exp",
            "attack_experiment_logger",
            "attack_instance_logger",
            "i0",
            0,
        ],
    ],
)
def test_is_curve_violation_rejects_unrelated_paths(path: list) -> None:
    """Errors outside the instance-metric path are never curve violations."""
    assert _is_curve_violation(_FakeError(path), {"attacks": {}}) is False


def test_is_curve_violation_handles_missing_lookup() -> None:
    """A well-shaped path with no matching value is not a curve violation."""
    path = [
        "attacks",
        "missing_exp",
        "attack_experiment_logger",
        "attack_instance_logger",
        "instance_0",
        "AUC",
    ]
    assert _is_curve_violation(_FakeError(path), {"attacks": {}}) is False


@pytest.mark.parametrize("payload", [[1, 2, 3], "a string", 42, None])
def test_non_dict_top_level_yields_empty_report(payload: object) -> None:
    """A non-object top-level report converts to empty attacks with a warning."""
    result = convert_report(payload, validate=False)
    assert result.report["attacks"] == {}
    assert any("not an object" in w for w in result.warnings)


def test_cli_convert_report_malformed_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI reports a friendly error and exits 1 on malformed JSON input."""
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    out = tmp_path / "out.json"
    monkeypatch.setattr(
        "sys.argv",
        ["sacroml", "convert-report", str(bad), str(out)],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "Could not parse" in capsys.readouterr().out
    assert not out.exists()


def test_cli_convert_report_missing_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI exits non-zero when the input file does not exist."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "sacroml",
            "convert-report",
            str(tmp_path / "nope.json"),
            str(tmp_path / "out.json"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_summary_dict_counts(edge_result: ConversionResult) -> None:
    """``summary_dict`` reduces the result to plain counts."""
    summary = edge_result.summary_dict()
    assert summary["is_valid"] is True
    assert summary["warnings"] == len(edge_result.warnings)
    assert summary["curve_warnings"] == len(edge_result.curve_warnings)
    assert summary["schema_errors"] == 0
    for dim, counts in summary["coverage"].items():
        assert counts["covered"] == len(edge_result.coverage[dim]["covered"])
        assert counts["missing"] == len(edge_result.coverage[dim]["missing"])


def test_summary_text_includes_sections(edge_result: ConversionResult) -> None:
    """``summary`` text surfaces coverage, warnings, curve notices and validity."""
    text = edge_result.summary()
    assert "metrics:" in text
    assert "Warnings (" in text
    assert "Curve-array notices (" in text
    assert "Converted report is schema-valid." in text


def test_summary_text_when_validation_skipped(
    edge_result: ConversionResult,
) -> None:
    """``validated=False`` suppresses the trailing schema-valid line."""
    text = edge_result.summary(validated=False)
    assert "schema-valid" not in text


def test_summary_text_for_schema_errors() -> None:
    """A schema-invalid result includes the NOT schema-valid footer."""
    legacy = {
        "LiRA Attack_zz": {
            "log_id": "zz",
            "log_time": "now",
            "metadata": {
                "sacroml_version": "1.0",
                "attack_name": "LiRA Attack",
                "attack_params": {},
                "global_metrics": {"AUC": 0.5},
            },
            "attack_experiment_logger": {
                "attack_instance_logger": {
                    "instance_0": {"AUC": {"unexpected": "object"}}
                }
            },
        }
    }
    result = convert_report(legacy)
    text = result.summary()
    assert "Schema errors (" in text
    assert "NOT schema-valid" in text


def test_renamed_instance_keys_are_compact() -> None:
    """All non-conforming keys get consecutive instance_<n> slots."""
    legacy = {
        "LiRA Attack_zz": {
            "log_id": "zz",
            "log_time": "now",
            "metadata": {
                "sacroml_version": "1.0",
                "attack_name": "LiRA Attack",
                "attack_params": {},
                "global_metrics": {"AUC": 0.5},
            },
            "attack_experiment_logger": {
                "attack_instance_logger": {
                    "first": {"AUC": 0.1},
                    "second": {"AUC": 0.2},
                    "third": {"AUC": 0.3},
                }
            },
        }
    }
    result = convert_report(legacy)
    logger = result.report["attacks"]["LiRA Attack_zz"]["attack_experiment_logger"][
        "attack_instance_logger"
    ]
    assert list(logger) == ["instance_0", "instance_1", "instance_2"]
