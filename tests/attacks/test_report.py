"""Test attack report module."""

from __future__ import annotations

import numpy as np
from fpdf import FPDF

from aisdc.attacks import report

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
    report._write_dict(  # pylint:disable=protected-access
        pdf, mydict, border=BORDER
    )
    pdf.close()
