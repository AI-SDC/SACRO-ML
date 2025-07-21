"""Utility functions for attacks."""

import contextlib

import numpy as np
from scipy.stats import shapiro

EPS: float = 1e-16  # Used to avoid numerical issues


def get_p_normal(samples: np.ndarray) -> float:
    """Test whether a set of samples is normally distributed."""
    p_normal: float = np.nan
    if np.nanvar(samples) > EPS:
        with contextlib.suppress(ValueError):
            _, p_normal = shapiro(samples)
    return p_normal
