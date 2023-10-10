"""Jim Smith October 2022
tests to pick up odd cases not otherwise covered
in code in the attacks folder.
"""

from __future__ import annotations

import numpy as np
import pytest
from fpdf import FPDF

from aisdc.attacks import attack, report
from aisdc.attacks.target import Target
from aisdc.safemodel.classifiers import SafeDecisionTreeClassifier

BORDER = 0


def test_superclass():
    """Test that the exceptions are raised if the superclass is called in error."""
    model = SafeDecisionTreeClassifier()
    target = Target(model=model)
    my_attack = attack.Attack()
    with pytest.raises(NotImplementedError):
        my_attack.attack(target)
    with pytest.raises(NotImplementedError):
        print(str(my_attack))  # .__str__()


def test_NumpyArrayEncoder():
    """Conversion routine
    from reports.py.
    """

    i32 = np.int32(2)
    i64 = np.int64(2)
    twoDarray = np.zeros((2, 2))
    my_encoder = report.NumpyArrayEncoder()

    retval = my_encoder.default(i32)
    assert isinstance(retval, int)

    retval = my_encoder.default(i64)
    assert isinstance(retval, int)

    retval = my_encoder.default(twoDarray)
    assert isinstance(retval, list)

    with pytest.raises(TypeError):
        retval = my_encoder.default("a string")


def test_line():
    """Code from report.py."""
    pdf = FPDF()
    pdf.add_page()
    report.line(pdf, "foo")
    pdf.close()


def test_dict():
    """Code from report.py."""
    pdf = FPDF()
    pdf.add_page()
    mydict = {"a": "hello", "b": "world"}
    report._write_dict(  # pylint:disable=protected-access
        pdf, mydict, indent=0, border=BORDER
    )
    pdf.close()
