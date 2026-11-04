"""Tests for per-record membership-inference signals.

Note: These were generated with Claude Sonnet 5.5.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from sacroml.attacks.model_sklearn import SklearnModel
from sacroml.attacks.signal import CLASSIFICATION_SIGNALS, get_signals
from sacroml.attacks.utils import EPS, logit

N = 120


@pytest.fixture(scope="module")
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    """Binary classification data."""
    return make_classification(
        n_samples=N, n_features=6, n_informative=4, n_classes=2, random_state=0
    )


@pytest.fixture(scope="module")
def multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    """Three-class classification data."""
    return make_classification(
        n_samples=N, n_features=8, n_informative=5, n_classes=3, random_state=0
    )


@pytest.fixture(scope="module")
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    """Noisy single-output regression data."""
    return make_regression(n_samples=N, n_features=5, noise=10.0, random_state=0)


def _clf(X: np.ndarray, y: np.ndarray) -> SklearnModel:
    """Return a fitted wrapped classifier."""
    model = LogisticRegression(max_iter=1000).fit(X, y)
    return SklearnModel(model, model_task="classifier")


def _reg(model, X: np.ndarray, y: np.ndarray) -> SklearnModel:
    """Return a fitted wrapped regressor."""
    return SklearnModel(model.fit(X, y), model_task="regressor")


@pytest.mark.parametrize("data_fixture", ["binary_data", "multiclass_data"])
def test_logit_matches_manual_calculation(data_fixture, request):
    """Signal equals logit(p_y) computed by hand from predict_proba."""
    X, y = request.getfixturevalue(data_fixture)
    model = _clf(X, y)

    result = get_signals(model, X, y)

    proba = model.predict_proba(X)
    p_y = proba[np.arange(len(y)), y]
    np.testing.assert_allclose(result, np.log(p_y / (1 - p_y)))


def test_logit_shape_and_dtype(multiclass_data):
    """One float value per record."""
    X, y = multiclass_data
    result = get_signals(_clf(X, y), X, y)
    assert result.shape == (len(y),)
    assert result.dtype == float
    assert np.isfinite(result).all()


def test_default_equals_explicit_logit(binary_data):
    """'default' resolves to 'logit' for classifiers."""
    X, y = binary_data
    model = _clf(X, y)
    np.testing.assert_array_equal(
        get_signals(model, X, y),
        get_signals(model, X, y, signal="logit"),
    )


def test_logit_higher_for_correct_than_wrong_label(binary_data):
    """Correct labels are more member-like than flipped ones."""
    X, y = binary_data
    model = _clf(X, y)
    correct = get_signals(model, X, y)
    flipped = get_signals(model, X, 1 - y)
    assert np.mean(correct) > np.mean(flipped)
    np.testing.assert_allclose(correct, -flipped)


def test_logit_absent_class_uses_logit_zero(multiclass_data):
    """Records of a class the model never saw get logit(0)."""
    X, y = multiclass_data
    seen = y != 2  # shadow model trained without class 2
    model = _clf(X[seen], y[seen])
    assert list(model.get_classes()) == [0, 1]

    result = get_signals(model, X, y)

    assert np.all(result[y == 2] == logit(0))
    assert np.all(result[y != 2] > logit(0))


def test_logit_supports_string_labels(binary_data):
    """Real (non-integer) class values are looked up via get_classes()."""
    X, y = binary_data
    y_str = np.where(y == 0, "no", "yes")
    model = _clf(X, y_str)

    result = get_signals(model, X, y_str)

    proba = model.predict_proba(X)
    col = np.array([list(model.get_classes()).index(label) for label in y_str])
    p_y = proba[np.arange(len(y_str)), col]
    np.testing.assert_allclose(result, np.log(p_y / (1 - p_y)))


def test_logit_rejects_positional_indices_for_non_range_classes(binary_data):
    """Indices that aren't real class values are treated as absent classes.

    Documents the label convention: passing positional indices for classes
    such as {10, 20} silently yields logit(0) for every record.
    """
    X, y = binary_data
    y_real = np.where(y == 0, 10, 20)
    model = _clf(X, y_real)

    result = get_signals(model, X, y)  # y holds 0/1, not 10/20

    assert np.all(result == logit(0))


def test_logit_length_mismatch_raises(binary_data):
    """Mismatched X and y lengths are an error, not silently truncated."""
    X, y = binary_data
    model = _clf(X, y)
    with pytest.raises(ValueError, match="shorter"):
        get_signals(model, X, y[:-1])


def test_log_mse_matches_manual_calculation(regression_data):
    """Signal equals -log(squared error + EPS) per record."""
    X, y = regression_data
    model = _reg(LinearRegression(), X, y)

    result = get_signals(model, X, y)

    sq_err = (model.predict(X) - y) ** 2
    np.testing.assert_allclose(result, -np.log(sq_err + EPS))


def test_log_mse_shape_and_dtype(regression_data):
    """One float value per record."""
    X, y = regression_data
    result = get_signals(_reg(LinearRegression(), X, y), X, y)
    assert result.shape == (len(y),)
    assert result.dtype == float


def test_default_equals_explicit_log_mse(regression_data):
    """'default' resolves to 'log_mse' for regressors."""
    X, y = regression_data
    model = _reg(LinearRegression(), X, y)
    np.testing.assert_array_equal(
        get_signals(model, X, y),
        get_signals(model, X, y, signal="log_mse"),
    )


def test_log_mse_larger_for_smaller_error(regression_data):
    """Signal is monotonically decreasing in the per-record error."""
    X, y = regression_data
    model = _reg(LinearRegression(), X, y)
    result = get_signals(model, X, y)
    err = (model.predict(X) - y) ** 2
    order = np.argsort(err)
    assert np.all(np.diff(result[order]) <= 0)


def test_log_mse_exact_fit_is_capped_by_eps(regression_data):
    """A perfect fit gives -log(EPS) rather than infinity."""
    X, y = regression_data
    model = _reg(DecisionTreeRegressor(random_state=0), X, y)
    result = get_signals(model, X, y)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, -np.log(EPS))


def test_log_mse_multi_output():
    """Multi-output errors are averaged across outputs per record."""
    X, y = make_regression(
        n_samples=N, n_features=5, n_targets=3, noise=5.0, random_state=1
    )
    model = _reg(LinearRegression(), X, y)

    result = get_signals(model, X, y)

    mse = np.mean((model.predict(X) - y) ** 2, axis=1)
    assert result.shape == (N,)
    np.testing.assert_allclose(result, -np.log(mse + EPS))


def test_unknown_signal_raises_for_classifier(binary_data):
    """Unknown names raise and list the available signals."""
    X, y = binary_data
    with pytest.raises(ValueError, match=r"Unknown signal 'nope'.*logit"):
        get_signals(_clf(X, y), X, y, signal="nope")


def test_unknown_signal_raises_for_regressor(regression_data):
    """Unknown names raise and list the available signals."""
    X, y = regression_data
    model = _reg(LinearRegression(), X, y)
    with pytest.raises(ValueError, match=r"Unknown signal 'nope'.*log_mse"):
        get_signals(model, X, y, signal="nope")


def test_signal_names_are_task_specific(binary_data, regression_data):
    """A regression signal can't be requested from a classifier and vice versa."""
    Xc, yc = binary_data
    Xr, yr = regression_data
    with pytest.raises(ValueError, match="Unknown signal"):
        get_signals(_clf(Xc, yc), Xc, yc, signal="log_mse")
    with pytest.raises(ValueError, match="Unknown signal"):
        get_signals(_reg(LinearRegression(), Xr, yr), Xr, yr, signal="logit")


def test_unsupported_task_raises():
    """Tasks without a registry (e.g. clusterer) are rejected."""
    X = np.random.default_rng(0).normal(size=(20, 3))
    model = SklearnModel(KMeans(n_clusters=2, n_init=1).fit(X), model_task="clusterer")
    with pytest.raises(ValueError, match="Unsupported 'model_task'"):
        get_signals(model, X, np.zeros(20))


def test_registered_signal_is_selectable(binary_data, monkeypatch):
    """A newly registered function is dispatched by name with (model, X, y)."""
    X, y = binary_data
    model = _clf(X, y)
    calls = []

    def constant_signal(m, X_, y_):
        calls.append((m, X_, y_))
        return np.full(len(y_), 3.0)

    monkeypatch.setitem(CLASSIFICATION_SIGNALS, "constant", constant_signal)

    result = get_signals(model, X, y, signal="constant")

    np.testing.assert_array_equal(result, np.full(len(y), 3.0))
    assert len(calls) == 1
    assert calls[0][0] is model
    # The default is unaffected by registering extra signals.
    np.testing.assert_array_equal(
        get_signals(model, X, y),
        get_signals(model, X, y, signal="logit"),
    )
