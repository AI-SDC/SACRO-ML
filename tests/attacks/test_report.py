"""Test attack report module."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
from fpdf import FPDF

from sacroml.attacks import report

BORDER = 0


def test_custom_json_encoder():
    """Test custom JSON encoder."""
    i32 = np.int32(2)
    i64 = np.int64(2)
    array_2d = np.zeros((2, 2))
    my_encoder = report.CustomJSONEncoder()

    retval = my_encoder.default(i32)
    assert isinstance(retval, int)

    retval = my_encoder.default(i64)
    assert isinstance(retval, int)

    retval = my_encoder.default(array_2d)
    assert isinstance(retval, list)


def test_line():
    """Test add line to PDF."""
    pdf = FPDF()
    pdf.add_page()
    report.line(pdf, "foo")
    pdf.close()


def test_dict():
    """Test write dictionary to PDF."""
    pdf = FPDF()
    pdf.add_page()
    mydict = {"a": "hello", "b": "world"}
    report._write_dict(pdf, mydict, border=BORDER)
    pdf.close()


def test_write_json_sanitises_non_finite_floats():
    """Test write_json replaces nan/inf values with null in JSON output.

    Parameters
    ----------
    None

    Notes
    -----
    Ensures that float('nan'), float('inf'), and float('-inf') are serialised
    as JSON null rather than the bare NaN/Infinity tokens that violate the
    JSON specification.
    """
    metrics = {
        "a_nan": float("nan"),
        "b_inf": float("inf"),
        "c_neginf": float("-inf"),
        "d_normal": 1.5,
        "BaNaNa": "BaNaNa",
        "array": np.array([1.0, np.nan, np.inf]),
    }
    output = {
        "metadata": {"attack_name": "test_attack"},
        "metrics": metrics,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = os.path.join(tmpdir, "report")
        report.write_json(output, dest)
        path = dest + ".json"
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)

    inner = data["test_attack"]["metrics"]
    assert inner["a_nan"] is None
    assert inner["b_inf"] is None
    assert inner["c_neginf"] is None
    assert inner["d_normal"] == pytest.approx(1.5)
    assert inner["BaNaNa"] == "BaNaNa"
    assert inner["array"] == [1.0, None, None]


def test_strip_keys():
    """Test _strip_keys removes specified keys without mutating the original."""
    original = {
        "metadata": {"attack_name": "test"},
        "instance_0": {
            "AUC": 0.8,
            "fpr": [0.0, 0.5, 1.0],
            "tpr": [0.0, 0.9, 1.0],
            "roc_thresh": [1.0, 0.5, 0.0],
        },
    }
    exclude = frozenset({"fpr", "tpr", "roc_thresh"})
    result = report._strip_keys(original, exclude)

    # Excluded keys are removed
    assert "fpr" not in result["instance_0"]
    assert "tpr" not in result["instance_0"]
    assert "roc_thresh" not in result["instance_0"]

    # Other keys are preserved
    assert result["instance_0"]["AUC"] == 0.8
    assert result["metadata"]["attack_name"] == "test"

    # Original dict is not mutated
    assert "fpr" in original["instance_0"]
    assert "tpr" in original["instance_0"]
    assert "roc_thresh" in original["instance_0"]


def test_write_json_excludes_large_arrays():
    """Test write_json strips excluded keys from JSON output."""
    metrics = {
        "AUC": 0.75,
        "fpr": np.linspace(0, 1, 100).tolist(),
        "tpr": np.linspace(0, 1, 100).tolist(),
        "roc_thresh": np.linspace(1, 0, 100).tolist(),
        "individual": {"score": [0.1, 0.9], "member": [0, 1]},
    }
    output = {
        "log_id": "abcdef1234567890",
        "metadata": {"attack_name": "test_attack"},
        "attack_experiment_logger": {
            "attack_instance_logger": {"instance_0": metrics},
        },
    }
    exclude = frozenset({"fpr", "tpr", "roc_thresh", "individual"})

    with tempfile.TemporaryDirectory() as tmpdir:
        dest = os.path.join(tmpdir, "report")
        report.write_json(output, dest, exclude_keys=exclude)
        with open(dest + ".json", encoding="utf-8") as fp:
            data = json.load(fp)

        # GenerateJSONModule keys the entry by "<attack_name>_<log_id>".
        attack_key = next(k for k in data if k.startswith("test_attack"))
        instance = data[attack_key]["attack_experiment_logger"][
            "attack_instance_logger"
        ]["instance_0"]
        assert "fpr" not in instance
        assert "tpr" not in instance
        assert "roc_thresh" not in instance
        assert "individual" not in instance
        assert instance["AUC"] == pytest.approx(0.75)

        # The stripped arrays are recoverable from the sidecar .npz.
        arrays_file = instance["arrays_file"]
        assert arrays_file.endswith("_instance_0.npz")
        with np.load(os.path.join(tmpdir, arrays_file)) as arrays:
            assert arrays["fpr"].shape == (100,)
            assert list(arrays["individual.member"]) == [0, 1]

    # write_json must not mutate the caller's in-memory output.
    assert (
        "fpr"
        in output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]
    )
