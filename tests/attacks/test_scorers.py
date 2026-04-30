"""Tests for sklearn-compatible MIA scorers in sacroml.attacks._scorers."""

from __future__ import annotations

import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from sacroml import metrics
from sacroml.attacks._scorers import SCORERS, _score_from_metrics, resolve_scorer


@pytest.fixture(name="fitted_rf")
def fixture_fitted_rf():
    """Return a small RandomForestClassifier fit on a synthetic 2-class problem."""
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=0,
    )
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X, y)
    return clf, X, y


@pytest.mark.parametrize("name", list(SCORERS.keys()))
def test_scorer_returns_float(fitted_rf, name):
    """Each SCORERS entry yields a float when called via the sklearn protocol."""
    clf, X, y = fitted_rf
    score = SCORERS[name](clf, X, y)
    assert isinstance(score, float)


def test_resolve_scorer_callable_passthrough():
    """Resolve_scorer returns the input unchanged when given a callable."""

    def my_scorer(*_args):
        return 0.0

    assert resolve_scorer(my_scorer) is my_scorer


def test_resolve_scorer_known_mia_name():
    """Resolve_scorer returns the SCORERS entry for a known MIA string."""
    assert resolve_scorer("AUC") is SCORERS["AUC"]


def test_resolve_scorer_sklearn_string_passthrough():
    """A recognised sklearn scoring string is returned unchanged."""
    assert resolve_scorer("roc_auc") == "roc_auc"


def test_resolve_scorer_unknown_string_raises():
    """An unrecognised string raises ValueError listing the MIA scorers."""
    with pytest.raises(ValueError, match="AUC"):
        resolve_scorer("not_a_real_scorer")


def test_score_from_metrics_bogus_key_raises(fitted_rf):
    """A bogus key raises KeyError whose message lists the available keys."""
    clf, X, y = fitted_rf
    with pytest.raises(KeyError, match="AUC"):
        _score_from_metrics(clf, X, y, key="DEFINITELY_NOT_A_KEY")


def test_scorer_value_matches_get_metrics(fitted_rf):
    """A SCORERS entry returns the same value as a direct get_metrics lookup."""
    clf, X, y = fitted_rf
    direct = metrics.get_metrics(clf.predict_proba(X), y)["AUC"]
    via_scorer = SCORERS["AUC"](clf, X, y)
    assert via_scorer == pytest.approx(float(direct))
