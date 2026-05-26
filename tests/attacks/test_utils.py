"""Tests for sacroml.attacks.utils helper functions."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sacroml.attacks.utils import unwrap_model


class TestUnwrapModel:
    """Tests for ``unwrap_model``."""

    def test_non_pipeline_returns_model_and_none(self):
        """A plain estimator is returned unchanged with no preprocessor."""
        model = SVC(gamma=0.1)
        estimator, preprocessor = unwrap_model(model)
        assert estimator is model
        assert preprocessor is None

    def test_single_step_pipeline_returns_final_step_only(self):
        """A one-step Pipeline yields its final estimator and no preprocessor."""
        final = LogisticRegression()
        pipe = Pipeline([("clf", final)])
        estimator, preprocessor = unwrap_model(pipe)
        assert estimator is final
        assert preprocessor is None

    def test_multi_step_pipeline_splits_preprocessor_from_estimator(self):
        """A multi-step Pipeline yields the final step and a Pipeline of the rest."""
        scaler = StandardScaler()
        final = LogisticRegression()
        pipe = Pipeline([("scaler", scaler), ("clf", final)])

        estimator, preprocessor = unwrap_model(pipe)

        assert estimator is final
        assert isinstance(preprocessor, Pipeline)
        assert [name for name, _ in preprocessor.steps] == ["scaler"]
        assert preprocessor.steps[0][1] is scaler

    def test_multi_step_preprocessor_transforms_input(self):
        """The returned preprocessor can transform inputs end-to-end."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 3))
        y = rng.integers(0, 2, size=20)

        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        pipe.fit(X, y)

        _, preprocessor = unwrap_model(pipe)
        transformed = preprocessor.transform(X)

        np.testing.assert_allclose(transformed.mean(axis=0), 0, atol=1e-8)
        np.testing.assert_allclose(transformed.std(axis=0), 1, atol=1e-1)
