"""Tests for legacy report.json -> new format conversion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sacroml.main import main
from sacroml.reporting import ConversionResult, convert_report, convert_report_file

FIXTURES = Path(__file__).parent / "fixtures"
DOCS_EXAMPLES = Path(__file__).parents[2] / "docs" / "source" / "attacks"


def _load(name: str) -> dict:
    """Load a fixture JSON file by name."""
    with open(FIXTURES / name, encoding="utf-8") as fh:
        return json.load(fh)


def _experiment(metadata: dict | None = None, instances: dict | None = None) -> dict:
    """Build a minimal, schema-valid legacy experiment for inline tests."""
    meta = {
        "sacroml_version": "1.4.0",
        "attack_name": "LiRA Attack",
        "attack_params": {},
        "global_metrics": {"AUC_sig": "Significant at p=0.05"},
    }
    if metadata is not None:
        meta.update(metadata)
    exp: dict = {"log_id": "id", "log_time": "now", "metadata": meta}
    if instances is not None:
        exp["attack_experiment_logger"] = {"attack_instance_logger": instances}
    return exp


@pytest.fixture
def basic_result() -> ConversionResult:
    """Conversion result for the well-formed legacy fixture."""
    return convert_report(_load("legacy_basic.json"))


# --- R1/R2: wrapping and catalog injection -------------------------------


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


# --- R3: schema validation -----------------------------------------------


def test_basic_conversion_is_schema_valid(basic_result: ConversionResult) -> None:
    """A clean legacy report converts to a schema-valid document."""
    assert basic_result.is_valid
    assert basic_result.schema_errors == []
    assert basic_result.warnings == []
    assert basic_result.curve_warnings == []


def test_conversion_is_idempotent(basic_result: ConversionResult) -> None:
    """Converting an already-converted report is a no-op for the payload."""
    again = convert_report(basic_result.report)
    assert again.report["attacks"] == basic_result.report["attacks"]
    assert again.is_valid


@pytest.mark.parametrize(
    "example",
    ["report_example_lira.json", "report_example_worstcase.json"],
)
def test_real_docs_examples_validate(example: str) -> None:
    """Real example reports convert, validate, and trip the load-bearing paths.

    Both real reports lack ``sacroml_version`` (injected) and contain
    ``roc_thresh``/curve arrays that start with ``null`` (downgraded to curve
    notices); without either behaviour the result would not be schema-valid.
    """
    with open(DOCS_EXAMPLES / example, encoding="utf-8") as fh:
        data = json.load(fh)
    result = convert_report(data)
    assert result.is_valid, result.schema_errors
    assert result.curve_warnings  # roc/curve null arrays present
    assert all(
        exp["metadata"]["sacroml_version"] == "unknown"
        for exp in result.report["attacks"].values()
    )


def test_non_curve_schema_error_is_reported() -> None:
    """An instance metric with an object value yields a real schema error."""
    legacy = {
        "LiRA Attack_yy": _experiment(
            instances={"instance_0": {"AUC": {"unexpected": "object"}}}
        )
    }
    result = convert_report(legacy)
    assert not result.is_valid
    assert result.schema_errors


def test_curve_array_is_warning_not_error() -> None:
    """An roc_thresh array starting with null is a curve notice, not an error."""
    legacy = {
        "LiRA Attack_zz": _experiment(
            instances={"instance_0": {"AUC": 0.75, "roc_thresh": [None, 1.0, 0.0]}}
        )
    }
    result = convert_report(legacy)
    assert result.is_valid
    assert result.curve_warnings
    assert all("roc_thresh" in w for w in result.curve_warnings)


# --- minimal normalisation (load-bearing on real reports) ----------------


def test_missing_sacroml_version_injected() -> None:
    """Reports without sacroml_version get an 'unknown' placeholder."""
    legacy = {"LiRA Attack_x": _experiment(metadata={"sacroml_version": None})}
    del legacy["LiRA Attack_x"]["metadata"]["sacroml_version"]
    result = convert_report(legacy)
    meta = result.report["attacks"]["LiRA Attack_x"]["metadata"]
    assert meta["sacroml_version"] == "unknown"
    assert any("sacroml_version was missing" in w for w in result.warnings)


def test_structural_attack_gets_empty_logger() -> None:
    """An instance-less attack gets an empty instance logger, not a crash."""
    legacy = {
        "Structural Attack_aaaa": {
            "log_id": "aaaa",
            "metadata": {
                "sacroml_version": "1.4.0",
                "attack_name": "Structural Attack",
                "attack_params": {},
                "global_metrics": {},
            },
        }
    }
    result = convert_report(legacy)
    struct = result.report["attacks"]["Structural Attack_aaaa"]
    assert struct["attack_experiment_logger"]["attack_instance_logger"] == {}
    assert result.is_valid


# --- R4: uncatalogued warnings -------------------------------------------


def test_basic_coverage_all_catalogued(basic_result: ConversionResult) -> None:
    """All metrics/params/attacks in the clean fixture are catalogued."""
    assert all(not missing for missing in basic_result.coverage.values())


def test_uncatalogued_entries_warn() -> None:
    """Unknown metrics/params/attacks are reported but not fatal."""
    legacy = {
        "Mystery attack_bbbb": _experiment(
            metadata={
                "attack_name": "Mystery attack",
                "attack_params": {"weird_param": True},
            },
            instances={"instance_0": {"AUC": 0.6, "weird_metric": 0.42}},
        )
    }
    result = convert_report(legacy)
    assert "weird_metric" in result.coverage["metrics"]
    assert "weird_param" in result.coverage["parameters"]
    assert "Mystery attack" in result.coverage["attacks"]
    assert any("Uncatalogued metric" in w for w in result.warnings)
    assert any("Uncatalogued parameter" in w for w in result.warnings)
    assert any("Uncatalogued attack" in w for w in result.warnings)


def test_no_validate_skips_validation() -> None:
    """Validate=False produces no schema errors but keeps coverage warnings."""
    legacy = {
        "Mystery attack_bbbb": _experiment(
            metadata={"attack_name": "Mystery attack"},
            instances={"instance_0": {"weird_metric": 0.42}},
        )
    }
    result = convert_report(legacy, validate=False)
    assert result.schema_errors == []
    assert result.curve_warnings == []
    assert result.coverage["metrics"]  # coverage is independent of validation


# --- robustness: non-object input must not crash (regression guard) ------


@pytest.mark.parametrize("payload", [[1, 2, 3], "a string", 42, None])
def test_non_dict_top_level_yields_empty_report(payload: object) -> None:
    """A non-object top-level report converts to empty attacks with a warning."""
    result = convert_report(payload, validate=False)
    assert result.report["attacks"] == {}
    assert any("not an object" in w for w in result.warnings)


# --- file + CLI entry points ---------------------------------------------


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
