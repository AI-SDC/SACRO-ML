"""Test the various models we have defined."""

from __future__ import annotations

import unittest

import numpy as np
import pytest
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sacroml.safemodel.classifiers import SafeSVC


def get_data():
    """Return data for testing."""
    cancer = datasets.load_breast_cancer()
    x = np.asarray(cancer["data"], dtype=np.float64)
    y = np.asarray(cancer["target"], dtype=np.float64)

    return x, y


class TestDPSVC(unittest.TestCase):
    """Test the differentially private SVC."""

    def test_run(self):
        """Test the model runs."""
        dpsvc = SafeSVC()
        svc = SVC(kernel="rbf", gamma="scale", C=1.0, probability=True)
        x, y = get_data()

        dpsvc.fit(x, y)
        svc.fit(x, y)

        test_features = x
        dp_predictions = dpsvc.predict(test_features)
        sv_predictions = svc.predict(test_features)

        # Check that the two models have equal shape
        assert dp_predictions.shape == sv_predictions.shape

        dp_predprob = dpsvc.predict_proba(test_features)
        sv_predprob = svc.predict_proba(test_features)

        # Check that the two models have equal shape
        assert dp_predprob.shape == sv_predprob.shape

    def test_svc_recommended(self):
        """Test using recommended values."""
        x, y = get_data()
        model = SafeSVC(gamma=1.0)
        model.fit(x, y)
        msg, disclosive = model.preliminary_check()
        correct_msg = "Model parameters are within recommended ranges.\n"
        assert msg == correct_msg
        assert not disclosive

    def test_svc_khat(self):
        """Test khat method."""
        x, y = get_data()
        model = SafeSVC(gamma=1.0)
        model.fit(x, y)
        y_matrix = np.ones((len(y), 30))
        assert len(y_matrix.shape) == 2
        _ = model.k_hat_svm(x, y_matrix)

    def test_svc_wrongdata(self):
        """Test with wrong datatypes."""
        x, y = get_data()
        model = SafeSVC(gamma=1.0)
        # wrong y datatype
        with pytest.raises(Exception, match="DPSCV needs np.ndarray inputs"):
            model.fit(x, 1)

        # wrong x datatype
        with pytest.raises(Exception, match="DPSCV needs np.ndarray inputs"):
            model.fit(1, 1)

        # wrong label values
        yplus = y + 5
        errstr = "DP SVC can only handle binary classification"
        with pytest.raises(Exception, match=errstr):
            model.fit(x, yplus)

    def test_svc_gamma_zero(self):
        """Test predictions if we provide daft params."""
        x, y = get_data()
        model = SafeSVC(gamma=0.0, eps=0.0)
        model.fit(x, y)
        predictions = model.predict(x)
        assert len(predictions) == len(x)

    def test_svc_gamma_auto(self):
        """Test predictions if we provide daft params."""
        x, y = get_data()
        model = SafeSVC(gamma="auto")
        model.fit(x, y)
        assert model.gamma == 1.0 / x.shape[1]

    def test_svc_setparams(self):
        """Test using unchanged values."""
        x, y = get_data()
        model = SafeSVC(gamma=1.0)
        model.fit(x, y)

        newvals = {"gamma": 5, "eps": 0, "dhat": 999}
        model.set_params(**newvals)
        model.fit(x, y)
        assert model.eps == 0
        assert model.lambdaval == 0
        assert model.dhat == 999
        assert model.gamma == 5

        # unknown
        unrecognised = {"foo": 99}
        model.set_params(**unrecognised)
        # should log to file mewssage "Unsupported parameter: foo"

    def test_svc_unchanged(self):
        """Test using unchanged values."""
        x, y = get_data()
        model = SafeSVC(gamma=1.0)
        model.fit(x, y)

        msg, disclosive = model.posthoc_check()
        correct_msg = ""
        assert msg == correct_msg
        assert not disclosive

    def test_svc_nonstd_params_changed_postfit(self):
        """Test with params changed after fit."""
        x, y = get_data()
        model = SafeSVC(gamma=1.0)
        model.fit(x, y)

        # change values
        model.svc = SVC().fit(x, y)  # make it disclosive
        model.platt_transform = LogisticRegression()
        msg, disclosive = model.posthoc_check()
        correct_msg = (
            "structure platt_transform has one difference.\n"
            "structure svc has several differences.\n"
        )

        assert msg == correct_msg
        assert disclosive
