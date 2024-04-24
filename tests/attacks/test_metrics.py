"""Test the metrics."""

from __future__ import annotations

import unittest

import numpy as np
import pytest

from aisdc.metrics import (
    _div,
    _tpr_at_fpr,
    get_metrics,
    get_probabilities,
    min_max_disc,
)

# pylint: disable = invalid-name

PREDICTED_CLASS = np.array([0, 1, 0, 0, 1, 1])
TRUE_CLASS = np.array([0, 0, 0, 1, 1, 1])
PREDICTED_PROBS = np.array(
    [[0.9, 0.1], [0.4, 0.6], [0.8, 0.2], [0.55, 0.45], [0.1, 0.9], [0.01, 0.99]]
)


class DummyClassifier:
    """Mocks the predict and predict_proba methods."""

    def predict(self, _):
        """Return dummy predictions."""
        return PREDICTED_CLASS

    def predict_proba(self, _):
        """Return dummy predicted probabilities."""
        return PREDICTED_PROBS


class TestInputExceptions(unittest.TestCase):
    """Test that the metrics.py errors with a helpful error message if an
    invalid shape is supplied.
    """

    def _create_fake_test_data(self):
        y_test = np.zeros(4)
        y_test[0] = 1
        return y_test

    def test_wrong_shape(self):
        """Test the check which ensures y_pred_proba is of shape [:,:]."""
        y_test = self._create_fake_test_data()
        with pytest.raises(ValueError):
            y_pred_proba = np.zeros((4, 2, 2))
            get_metrics(y_pred_proba, y_test)

    def test_wrong_size(self):
        """Test the check which ensures y_pred_proba is of size (:,2)."""
        y_test = self._create_fake_test_data()
        with pytest.raises(ValueError):
            y_pred_proba = np.zeros((4, 4))
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

        self.assertAlmostEqual(0.75, acc)
        self.assertAlmostEqual(0.5, auc)
        self.assertAlmostEqual(0.5, p_auc)
        self.assertAlmostEqual(0.0, tpr)


class TestProbabilities(unittest.TestCase):
    """Test the checks on the input parameters of the get_probabilites function."""

    def test_permute_rows_errors(self):
        """
        Test to make sure an error is thrown when permute_rows is set to True,
        but no y_test is supplied.
        """
        clf = DummyClassifier()
        testX = []

        with pytest.raises(ValueError):
            get_probabilities(clf, testX, permute_rows=True)

    def test_permute_rows_with_permute_rows(self):
        """Test permute_rows = True succeeds."""

        clf = DummyClassifier()
        testX = np.zeros((4, 2))
        testY = np.zeros((4, 2))

        returned = get_probabilities(clf, testX, testY, permute_rows=True)

        # Check the function returns two arguments
        self.assertEqual(2, len(returned))

        # Check that the second argument is the same shape as testY
        self.assertEqual(testY.shape, returned[1].shape)

        # Check that the function is returning the right thing: predict_proba
        self.assertEqual(clf.predict_proba(testX).shape, returned[0].shape)

    def test_permute_rows_without_permute_rows(self):
        """Test permute_rows = False succeeds."""

        clf = DummyClassifier()
        testX = np.zeros((4, 2))

        y_pred_proba = get_probabilities(clf, testX, permute_rows=False)

        # Check the function returns pnly y_pred_proba
        self.assertEqual(clf.predict_proba(testX).shape, y_pred_proba.shape)


class TestMetrics(unittest.TestCase):
    """Test the metrics with some dummy predictions."""

    def test_metrics(self):
        """Test each individual metric with dummy data."""
        clf = DummyClassifier()
        testX = []
        testy = TRUE_CLASS
        y_pred_proba = get_probabilities(clf, testX, testy, permute_rows=False)
        metrics = get_metrics(y_pred_proba, testy)
        self.assertAlmostEqual(metrics["TPR"], 2 / 3)
        self.assertAlmostEqual(metrics["FPR"], 1 / 3)
        self.assertAlmostEqual(metrics["FAR"], 1 / 3)
        self.assertAlmostEqual(metrics["TNR"], 2 / 3)
        self.assertAlmostEqual(metrics["PPV"], 2 / 3)
        self.assertAlmostEqual(metrics["NPV"], 2 / 3)
        self.assertAlmostEqual(metrics["FNR"], 1 / 3)
        self.assertAlmostEqual(metrics["ACC"], 4 / 6)
        self.assertAlmostEqual(metrics["F1score"], (8 / 9) / (2 / 3 + 2 / 3))
        self.assertAlmostEqual(metrics["Advantage"], 1 / 3)
        self.assertAlmostEqual(metrics["AUC"], 8 / 9)

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
        self.assertEqual(-115.13, pval)

        # wrong predictions - probaility very close to 1 so logp=0
        _, _, _, pval = min_max_disc(y, wrong)
        self.assertAlmostEqual(0.0, pval)


class TestFPRatTPR(unittest.TestCase):
    """Test code that computes TPR at fixed FPR."""

    def test_tpr(self):
        """Test tpr at fpr."""
        y_true = TRUE_CLASS
        y_score = PREDICTED_PROBS[:, 1]

        tpr = _tpr_at_fpr(y_true, y_score, fpr=0)
        self.assertAlmostEqual(tpr, 2 / 3)
        tpr = _tpr_at_fpr(y_true, y_score, fpr=0.001)
        self.assertAlmostEqual(tpr, 2 / 3)
        tpr = _tpr_at_fpr(y_true, y_score, fpr=0.1)
        self.assertAlmostEqual(tpr, 2 / 3)
        tpr = _tpr_at_fpr(y_true, y_score, fpr=0.4)
        self.assertAlmostEqual(tpr, 1)
        tpr = _tpr_at_fpr(y_true, y_score, fpr=1.0)
        self.assertAlmostEqual(tpr, 1)
        tpr = _tpr_at_fpr(y_true, y_score, fpr_perc=True, fpr=100.0)
        self.assertAlmostEqual(tpr, 1)


class Test_Div(unittest.TestCase):
    """Tests the _div functionality."""

    def test_div(self):
        """Test div for y=1 and 0."""
        result = _div(8.0, 1.0, 99.0)
        self.assertAlmostEqual(result, 8.0)
        result2 = _div(8.0, 0.0, 99.0)
        self.assertAlmostEqual(result2, 99.0)


class TestExtreme(unittest.TestCase):
    """Test the extreme metrics."""

    def test_extreme_default(self):
        """Tets with the dummy data."""
        pred_probs = DummyClassifier().predict_proba(None)[:, 1]
        maxd, mind, mmd, _ = min_max_disc(TRUE_CLASS, pred_probs)

        # 10% of 6 is 1 so:
        # maxd should be 1 (the highest one is predicted as1)
        # mind should be 0 (the lowest one is not predicted as1)
        self.assertAlmostEqual(maxd, 1.0)
        self.assertAlmostEqual(mind, 0.0)
        self.assertAlmostEqual(mmd, 1.0)

    def test_extreme_higer_prop(self):
        """Tets with the dummy data but increase proportion to 0.5."""
        pred_probs = DummyClassifier().predict_proba(None)[:, 1]
        maxd, mind, mmd, _ = min_max_disc(TRUE_CLASS, pred_probs, x_prop=0.5)

        # 10% of 6 is 1 so:
        # maxd should be 1 (the highest one is predicted as1)
        # mind should be 0 (the lowest one is not predicted as1)
        self.assertAlmostEqual(maxd, 2 / 3)
        self.assertAlmostEqual(mind, 1 / 3)
        self.assertAlmostEqual(mmd, 1 / 3)
