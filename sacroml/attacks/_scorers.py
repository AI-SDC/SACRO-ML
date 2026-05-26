"""Sklearn-compatible scorers for MIA metrics from :func:`sacroml.metrics.get_metrics`.

These scorers conform to the standard sklearn scorer call protocol
``(estimator, X, y)`` and can be passed directly to ``GridSearchCV`` /
``RandomizedSearchCV`` via the ``scoring`` argument.

The metrics produced by :func:`sacroml.metrics.get_metrics` are richer than
what :func:`sklearn.metrics.make_scorer` can address through its standard
``score_func(y_true, y_pred)`` signature, so each scorer is implemented as a
``functools.partial`` over a single private function that runs
``predict_proba`` and pulls the requested key out of the metrics dict.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer_names

from sacroml import metrics

Scorer = Callable[[BaseEstimator, np.ndarray, np.ndarray], float]

_SCORER_KEYS: tuple[str, ...] = (
    "AUC",
    "TPR@0.1",
    "TPR@0.001",
    "FDIF01",
    "FDIF02",
    "Advantage",
    "PDIF01",
)


def _score_from_metrics(
    estimator: BaseEstimator,
    X: np.ndarray,
    y_true: np.ndarray,
    *,
    key: str,
) -> float:
    """Compute a single MIA metric via :func:`sacroml.metrics.get_metrics`.

    Conforms to the sklearn scorer call protocol ``(estimator, X, y)`` so it
    can be used directly (or via :func:`functools.partial`) as the ``scoring``
    argument of any sklearn search/CV object.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        A fitted estimator exposing ``predict_proba``.
    X : np.ndarray
        Feature matrix passed to ``estimator.predict_proba``.
    y_true : np.ndarray
        True labels (binary; the shape required by
        :func:`sacroml.metrics.get_metrics`).
    key : str
        Name of the metric to extract from the dict returned by
        :func:`sacroml.metrics.get_metrics`.

    Returns
    -------
    float
        The value at ``key`` cast to ``float``.

    Raises
    ------
    KeyError
        If ``key`` is not present in the metrics dict. The error message
        lists the keys that are available.
    """
    y_pred_proba = estimator.predict_proba(X)
    out = metrics.get_metrics(y_pred_proba, y_true)
    if key not in out:
        available = sorted(out.keys())
        msg = (
            f"Metric key {key!r} not found in get_metrics output. "
            f"Available keys: {available}."
        )
        raise KeyError(msg)
    return float(out[key])


SCORERS: dict[str, Scorer] = {
    name: partial(_score_from_metrics, key=name) for name in _SCORER_KEYS
}


def resolve_scorer(name: str | Scorer) -> str | Scorer:
    """Resolve a scorer specifier into a value usable as sklearn ``scoring``.

    Parameters
    ----------
    name : str or Callable
        - A callable is returned unchanged (assumed to be a user-supplied
          scorer following the ``(estimator, X, y)`` protocol).
        - A string in :data:`SCORERS` returns the corresponding partial.
        - A string accepted by sklearn (e.g. ``"roc_auc"``) is returned
          unchanged so sklearn resolves it at fit time.

    Returns
    -------
    str or Callable
        Either the original callable, the matching :data:`SCORERS` entry, or
        the input string passed through to sklearn.

    Raises
    ------
    ValueError
        If ``name`` is neither a callable, a known MIA scorer name, nor a
        recognised sklearn scoring string.
    """
    if callable(name):
        return name
    if not isinstance(name, str):
        msg = f"Scorer must be a callable or a string; got {type(name).__name__}."
        raise ValueError(msg)
    if name in SCORERS:
        return SCORERS[name]
    if name in get_scorer_names():
        return name
    available_mia = sorted(SCORERS.keys())
    msg = (
        f"Unknown scorer {name!r}. "
        f"Known MIA scorers: {available_mia}. "
        "Any sklearn scoring string (see sklearn.metrics.get_scorer_names()) "
        "is also valid."
    )
    raise ValueError(msg)
