"""Tests for safetf."""

from __future__ import annotations

import pytest

from sacroml.safemodel.classifiers import safetf


def test_safe_tf_dpmodel_l2_and_noise():
    """Test user is informed this is not implemented yet."""
    with pytest.raises(NotImplementedError):
        # with values for the l2 and noise params
        safetf.SafeTFModel(1.5, 2.0, True)
