"""Jim Smith 2022
    Not currently implemented.
"""

from __future__ import annotations

import pytest

from aisdc.safemodel.classifiers import safetf


def test_Safe_tf_DPModel_l2_and_noise():
    """Tests user is informed this is not implemented yet."""
    with pytest.raises(NotImplementedError):
        # with values for the l2 and noise params
        safetf.Safe_tf_DPModel(1.5, 2.0, True)
