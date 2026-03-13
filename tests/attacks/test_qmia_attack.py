"""Test QMIA attack."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sacroml.attacks.qmia_attack import QMIAAttack
from sacroml.attacks.target import Target
from sacroml.attacks.utils import (
    extract_true_label_probs,
    margins_to_two_column_probs,
    membership_labels,
    qmia_binary_score,
)

pytest.importorskip("catboost")


@pytest.fixture(name="qmia_binary_target")
def fixture_qmia_binary_target() -> Target:
    """Return a binary tabular target suitable for QMIA tests."""
    X, y = make_classification(
        n_samples=240,
        n_features=8,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        class_sep=1.25,
        random_state=7,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=7
    )

    model = RandomForestClassifier(n_estimators=50, random_state=7)
    model.fit(X_train, y_train)

    target = Target(
        model=model,
        dataset_name="qmia_binary",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_train_orig=X_train,
        y_train_orig=y_train,
        X_test_orig=X_test,
        y_test_orig=y_test,
    )
    for idx in range(X.shape[1]):
        target.add_feature(f"V{idx}", [idx], "float")
    return target


@pytest.fixture(name="qmia_multiclass_target")
def fixture_qmia_multiclass_target() -> Target:
    """Return a multiclass target rejected by the binary-only QMIA v1 path."""
    X, y = make_classification(
        n_samples=180,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=9,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=9
    )

    model = RandomForestClassifier(n_estimators=40, random_state=9)
    model.fit(X_train, y_train)

    return Target(
        model=model,
        dataset_name="qmia_multiclass",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_train_orig=X_train,
        y_train_orig=y_train,
        X_test_orig=X_test,
        y_test_orig=y_test,
    )


def test_extract_true_label_probs():
    """True-label probability extraction should follow the label indices."""
    probas = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    labels = np.array([0, 1, 0])

    values = extract_true_label_probs(probas, labels)

    np.testing.assert_allclose(values, np.array([0.8, 0.7, 0.6]))


def test_qmia_binary_score():
    """QMIA binary score should equal the logit of the true-label probability."""
    probas = np.array([[0.8, 0.2], [0.3, 0.7]])
    labels = np.array([0, 1])

    scores = qmia_binary_score(probas, labels)

    np.testing.assert_allclose(
        scores,
        np.array([np.log(0.8 / 0.2), np.log(0.7 / 0.3)]),
    )


def test_membership_labels():
    """Membership labels should mark train rows before test rows."""
    np.testing.assert_array_equal(membership_labels(3, 2), np.array([1, 1, 1, 0, 0]))


def test_margins_to_two_column_probs():
    """QMIA margin conversion should preserve ordering and a 2-column shape."""
    margins = np.array([-2.0, 0.0, 2.0])

    probs = margins_to_two_column_probs(margins)

    assert probs.shape == (3, 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(3))
    assert probs[0, 1] < probs[1, 1] < probs[2, 1]


def test_qmia_insufficient_target_returns_empty_report(tmp_path):
    """QMIA should no-op when required target details are missing."""
    attack_obj = QMIAAttack(output_dir=str(tmp_path), write_report=False)
    output = attack_obj.attack(Target())
    assert not output


def test_qmia_runs_on_binary_tabular_target(qmia_binary_target, tmp_path):
    """QMIA should produce a standard attack report on a valid target."""
    attack_obj = QMIAAttack(
        output_dir=str(tmp_path / "qmia"),
        write_report=False,
        catboost_params={"iterations": 20, "depth": 3},
    )

    output = attack_obj.attack(qmia_binary_target)

    assert output["metadata"]["attack_name"] == "QMIA Attack"
    metrics = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]
    assert 0 <= metrics["TPR"] <= 1
    assert 0 <= metrics["FPR"] <= 1
    assert 0 <= metrics["AUC"] <= 1


def test_qmia_metadata_contains_alpha_and_mode(qmia_binary_target, tmp_path):
    """QMIA metadata should expose the main attack knobs."""
    attack_obj = QMIAAttack(
        output_dir=str(tmp_path / "qmia"),
        write_report=False,
        alpha=0.1,
        use_gaussian=True,
        catboost_params={"iterations": 20, "depth": 3},
    )

    output = attack_obj.attack(qmia_binary_target)
    metadata = output["metadata"]

    assert metadata["attack_params"]["alpha"] == 0.1
    assert metadata["attack_params"]["use_gaussian"]
    assert metadata["global_metrics"]["regressor_mode"] == "gaussian_uncertainty"
    assert metadata["global_metrics"]["qmia_score"] == "binary_true_label_logit"
    assert metadata["global_metrics"]["public_slice"] == "target.X_test"


def test_qmia_attack_instance_logger_shape(qmia_binary_target, tmp_path):
    """QMIA output should preserve the standard instance logger schema."""
    attack_obj = QMIAAttack(
        output_dir=str(tmp_path / "qmia"),
        write_report=False,
        report_individual=True,
        catboost_params={"iterations": 20, "depth": 3},
    )

    output = attack_obj.attack(qmia_binary_target)
    instance = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]

    assert "TPR" in instance
    assert "FPR" in instance
    assert "individual" in instance
    assert "member_prob" in instance["individual"]
    assert "threshold" in instance["individual"]
    assert "margin" in instance["individual"]


def test_qmia_use_gaussian_false_runs(qmia_binary_target, tmp_path):
    """QMIA should support the direct quantile fallback path."""
    attack_obj = QMIAAttack(
        output_dir=str(tmp_path / "qmia"),
        write_report=False,
        use_gaussian=False,
        catboost_params={"iterations": 20, "depth": 3},
    )

    output = attack_obj.attack(qmia_binary_target)
    metrics = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]

    assert output["metadata"]["global_metrics"]["regressor_mode"] == "direct_quantile"
    assert 0 <= metrics["AUC"] <= 1


def test_qmia_invalid_alpha_raises(qmia_binary_target, tmp_path):
    """QMIA should reject invalid alpha values."""
    attack_obj = QMIAAttack(output_dir=str(tmp_path), write_report=False, alpha=0.0)

    with pytest.raises(ValueError, match="alpha must lie strictly between 0 and 1"):
        attack_obj.attack(qmia_binary_target)


def test_qmia_non_binary_target_returns_empty_report(qmia_multiclass_target, tmp_path):
    """QMIA v1 should reject non-binary classification targets."""
    attack_obj = QMIAAttack(output_dir=str(tmp_path), write_report=False)

    output = attack_obj.attack(qmia_multiclass_target)

    assert not output


def test_qmia_public_fpr_tracks_alpha(qmia_binary_target, tmp_path):
    """QMIA should approximately control FPR on the public slice."""
    alpha = 0.2
    attack_obj = QMIAAttack(
        output_dir=str(tmp_path / "qmia"),
        write_report=False,
        alpha=alpha,
        catboost_params={"iterations": 25, "depth": 3},
    )

    output = attack_obj.attack(qmia_binary_target)
    metrics = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]

    assert abs(metrics["observed_public_fpr"] - alpha) < 0.2
