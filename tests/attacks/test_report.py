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
