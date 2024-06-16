"""Test the metrics."""

from __future__ import annotations

import unittest

import numpy as np
import pytest

from aisdc.metrics import _div, _tpr_at_fpr, get_metrics, min_max_disc

PREDICTED_CLASS = np.array([0, 1, 0, 0, 1, 1])
TRUE_CLASS = np.array([0, 0, 0, 1, 1, 1])
PREDICTED_PROBS = np.array(
    [[0.9, 0.1], [0.4, 0.6], [0.8, 0.2], [0.55, 0.45], [0.1, 0.9], [0.01, 0.99]]
)


class DummyClassifier:
    """Mock the predict and predict_proba methods."""

    def predict(self, _):
        """Return dummy predictions."""
        return PREDICTED_CLASS

    def predict_proba(self, _):
        """Return dummy predicted probabilities."""
        return PREDICTED_PROBS


class TestInputExceptions(unittest.TestCase):
    """Test error message if an invalid shape is supplied."""

    def _create_fake_test_data(self):
        y_test = np.zeros(4)
        y_test[0] = 1
        return y_test

    def test_wrong_shape(self):
        """Test the check which ensures y_pred_proba is of shape [:,:]."""
        y_test = self._create_fake_test_data()
        y_pred_proba = np.zeros((4, 2, 2))
        with pytest.raises(ValueError, match="y_pred.*"):
            get_metrics(y_pred_proba, y_test)

    def test_wrong_size(self):
        """Test the check which ensures y_pred_proba is of size (:,2)."""
        y_test = self._create_fake_test_data()
        y_pred_proba = np.zeros((4, 4))
        with pytest.raises(ValueError, match=".*multiclass.*"):
            get_metrics(y_pred_proba, y_test)

    def test_valid_input(self):
        """Test to make sure a valid array does not throw an exception."""
        y_test = self._create_fake_test_data()
        y_pred_proba = np.zeros((4, 2))
        returned = get_metrics(y_pred_proba, y_test)
        acc = returned["ACC"]
        auc = returned["AUC"]
        p_auc = returned["P_HIGHER_AUC"]
        tpr = returned["TPR"]
        assert pytest.approx(acc) == 0.75
        assert pytest.approx(auc) == 0.5
        assert pytest.approx(p_auc) == 0.5
        assert pytest.approx(tpr) == 0.0


class TestMetrics(unittest.TestCase):
    """Test the metrics with some dummy predictions."""

    def test_metrics(self):
        """Test each individual metric with dummy data."""
        clf = DummyClassifier()
        X_test = []
        y_test = TRUE_CLASS
        y_pred_proba = clf.predict_proba(X_test)
        metrics = get_metrics(y_pred_proba, y_test)
        assert metrics["TPR"] == pytest.approx(2 / 3)
        assert metrics["FPR"] == pytest.approx(1 / 3)
        assert metrics["FAR"] == pytest.approx(1 / 3)
        assert metrics["TNR"] == pytest.approx(2 / 3)
        assert metrics["PPV"] == pytest.approx(2 / 3)
        assert metrics["NPV"] == pytest.approx(2 / 3)
        assert metrics["FNR"] == pytest.approx(1 / 3)
        assert metrics["ACC"] == pytest.approx(4 / 6)
        assert metrics["F1score"] == pytest.approx((8 / 9) / (2 / 3 + 2 / 3))
        assert metrics["Advantage"] == pytest.approx(1 / 3)
        assert metrics["AUC"] == pytest.approx(8 / 9)

    def test_mia_extremecase(self):
        """Test the extreme case mia in metrics.py."""
        # create actual values
        y = np.zeros(50000)
        y[:25] = 1
        # exactly right and wrong predictions
        right = np.zeros(50000)
        right[:25] = 1
        wrong = 1 - right

        # right predictions - triggers override for very small logp
        _, _, _, pval = min_max_disc(y, right)
        assert pval == -115.13

        # wrong predictions - probaility very close to 1 so logp=0
        _, _, _, pval = min_max_disc(y, wrong)
        assert pytest.approx(pval) == 0


class TestFPRatTPR(unittest.TestCase):
    """Test code that computes TPR at fixed FPR."""

    def test_tpr(self):
        """Test tpr at fpr."""
        y_true = TRUE_CLASS
        y_score = PREDICTED_PROBS[:, 1]
        tpr = _tpr_at_fpr(y_true, y_score, fpr=0)
        assert tpr == pytest.approx(2 / 3)
        tpr = _tpr_at_fpr(y_true, y_score, fpr=0.001)
        assert tpr == pytest.approx(2 / 3)
        tpr = _tpr_at_fpr(y_true, y_score, fpr=0.1)
        assert tpr == pytest.approx(2 / 3)
        tpr = _tpr_at_fpr(y_true, y_score, fpr=0.4)
        assert tpr == pytest.approx(1)
        tpr = _tpr_at_fpr(y_true, y_score, fpr=1.0)
        assert tpr == pytest.approx(1)
        tpr = _tpr_at_fpr(y_true, y_score, fpr_perc=True, fpr=100.0)
        assert tpr == pytest.approx(1)


class TestDiv(unittest.TestCase):
    """Test the _div functionality."""

    def test_div(self):
        """Test div for y=1 and 0."""
        result = _div(8.0, 1.0, 99.0)
        assert result == pytest.approx(8.0)
        result2 = _div(8.0, 0.0, 99.0)
        assert result2 == pytest.approx(99.0)


class TestExtreme(unittest.TestCase):
    """Test the extreme metrics."""

    def test_extreme_default(self):
        """Tets with the dummy data."""
        pred_probs = DummyClassifier().predict_proba(None)[:, 1]
        maxd, mind, mmd, _ = min_max_disc(TRUE_CLASS, pred_probs)

        # 10% of 6 is 1 so:
        # maxd should be 1 (the highest one is predicted as1)
        # mind should be 0 (the lowest one is not predicted as1)
        assert maxd == pytest.approx(1)
        assert mind == pytest.approx(0)
        assert mmd == pytest.approx(1)

    def test_extreme_higer_prop(self):
        """Tets with the dummy data but increase proportion to 0.5."""
        pred_probs = DummyClassifier().predict_proba(None)[:, 1]
        maxd, mind, mmd, _ = min_max_disc(TRUE_CLASS, pred_probs, x_prop=0.5)

        # 10% of 6 is 1 so:
        # maxd should be 1 (the highest one is predicted as1)
        # mind should be 0 (the lowest one is not predicted as1)
        assert maxd == pytest.approx(2 / 3)
        assert mind == pytest.approx(1 / 3)
        assert mmd == pytest.approx(1 / 3)
