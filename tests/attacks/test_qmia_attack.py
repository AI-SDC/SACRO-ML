"""Test QMIA attack."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

from sacroml.attacks.qmia_attack import QMIAAttack
from sacroml.attacks.target import Target
from sacroml.attacks.utils import (
    margins_to_two_column_probs,
    membership_labels,
    qmia_hinge_score,
)


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
    """Return a multiclass target for QMIA."""
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


def test_qmia_hinge_score():
    """QMIA hinge score should equal logit(p_y) - max_{y'!=y} logit(p_{y'})."""
    probas = np.array([[0.8, 0.2], [0.3, 0.7]])
    labels = np.array([0, 1])

    scores = qmia_hinge_score(probas, labels)

    # For binary: logit(p_y) - logit(1-p_y) = 2 * logit(p_y)
    np.testing.assert_allclose(
        scores,
        np.array([2 * np.log(0.8 / 0.2), 2 * np.log(0.7 / 0.3)]),
    )


def test_qmia_hinge_score_multiclass():
    """QMIA hinge score should work for multiclass."""
    probas = np.array([[0.2, 0.5, 0.3]])
    labels = np.array([1])

    scores = qmia_hinge_score(probas, labels)

    # logit(0.5) - max(logit(0.2), logit(0.3)) = logit(0.5) - logit(0.3)
    expected = np.log(0.5 / 0.5) - np.log(0.3 / 0.7)
    np.testing.assert_allclose(scores, [expected])


def test_membership_labels():
    """Membership labels should mark train rows before test rows."""
    np.testing.assert_array_equal(membership_labels(3, 2), np.array([1, 1, 1, 0, 0]))


def test_margins_to_two_column_probs():
    """QMIA margin conversion should preserve ordering and a 2-column shape."""
    margins: np.ndarray = np.array([-2.0, 0.0, 2.0])

    probs: np.ndarray = margins_to_two_column_probs(margins)

    assert probs.shape == (3, 2)
    # Column 0 is the non-member score (negated margin), column 1 the member score.
    np.testing.assert_allclose(probs[:, 0], -margins)
    np.testing.assert_allclose(probs[:, 1], margins)
    # argmax selects column 1 iff margin > 0 (member prediction).
    np.testing.assert_array_equal(np.argmax(probs, axis=1), np.array([0, 0, 1]))


def test_margins_to_two_column_probs_preserves_tail_order():
    """Large margins must remain rank-distinguishable (no sigmoid saturation)."""
    margins: np.ndarray = np.array([100.0, 200.0, 300.0])

    probs: np.ndarray = margins_to_two_column_probs(margins)

    # With a sigmoid+float64 clip, all three would collapse to 1.0 and tie.
    assert probs[0, 1] < probs[1, 1] < probs[2, 1]
    assert len(np.unique(probs[:, 1])) == 3


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
    )

    output = attack_obj.attack(qmia_binary_target)

    assert output["metadata"]["attack_name"] == "QMIA Attack"
    m = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]
    assert 0 <= m["TPR"] <= 1
    assert 0 <= m["FPR"] <= 1
    assert 0 <= m["AUC"] <= 1


def test_qmia_metadata_contains_alpha_and_mode(qmia_binary_target, tmp_path):
    """QMIA metadata should expose the main attack knobs."""
    attack_obj = QMIAAttack(
        output_dir=str(tmp_path / "qmia"),
        write_report=False,
        alpha=0.1,
    )

    output = attack_obj.attack(qmia_binary_target)
    metadata = output["metadata"]

    assert metadata["attack_params"]["alpha"] == 0.1
    assert "AUC_sig" in metadata["global_metrics"]
    assert "TPR" in metadata["global_metrics"]


def test_qmia_attack_instance_logger_shape(qmia_binary_target, tmp_path):
    """QMIA output should preserve the standard instance logger schema."""
    attack_obj = QMIAAttack(
        output_dir=str(tmp_path / "qmia"),
        write_report=False,
        report_individual=True,
    )

    output = attack_obj.attack(qmia_binary_target)
    instance_logger = output["attack_experiment_logger"]["attack_instance_logger"]
    instance = instance_logger["instance_0"]

    assert "TPR" in instance
    assert "FPR" in instance
    assert "individual" in instance
    assert "member_prob" in instance["individual"]
    assert "threshold" in instance["individual"]
    assert "margin" in instance["individual"]


def test_qmia_invalid_alpha_raises(qmia_binary_target, tmp_path):
    """QMIA should reject invalid alpha values."""
    attack_obj = QMIAAttack(output_dir=str(tmp_path), write_report=False, alpha=0.0)

    with pytest.raises(ValueError, match="alpha must lie strictly between 0 and 1"):
        attack_obj.attack(qmia_binary_target)


def test_qmia_multiclass_target_runs(qmia_multiclass_target, tmp_path):
    """QMIA should handle multiclass classification targets."""
    attack_obj = QMIAAttack(
        output_dir=str(tmp_path / "qmia"),
        write_report=False,
    )

    output = attack_obj.attack(qmia_multiclass_target)

    assert output
    m = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]
    assert 0 <= m["AUC"] <= 1


def test_qmia_public_fpr_tracks_alpha(qmia_binary_target, tmp_path):
    """QMIA should approximately control FPR on the public slice."""
    alpha = 0.2
    attack_obj = QMIAAttack(
        output_dir=str(tmp_path / "qmia"),
        write_report=False,
        alpha=alpha,
    )

    output = attack_obj.attack(qmia_binary_target)
    m = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]

    assert abs(m["observed_public_fpr"] - alpha) < 0.2


def test_qmia_get_params_includes_p_thresh():
    """Get_params should return all constructor arguments including p_thresh."""
    attack_obj = QMIAAttack(alpha=0.05, p_thresh=0.01, max_iter=50)
    params = attack_obj.get_params()
    assert params["alpha"] == 0.05
    assert params["p_thresh"] == 0.01
    assert params["max_iter"] == 50
    assert "output_dir" in params
    assert "write_report" in params


def test_qmia_str():
    """__str__ should return 'QMIA Attack'."""
    assert str(QMIAAttack()) == "QMIA Attack"


def test_qmia_construct_metadata_global_metrics(qmia_binary_target, tmp_path):
    """_construct_metadata should populate AUC significance and key metrics."""
    attack_obj = QMIAAttack(
        output_dir=str(tmp_path / "qmia"),
        write_report=False,
        p_thresh=0.05,
    )

    output = attack_obj.attack(qmia_binary_target)
    gm = output["metadata"]["global_metrics"]

    assert "alpha" in gm
    assert "p_thresh" in gm
    assert gm["p_thresh"] == 0.05
    assert "AUC_sig" in gm
    assert "null_auc_3sd_range" in gm
    assert "TPR" in gm
    assert "FPR" in gm
    assert "Advantage" in gm


def test_qmia_make_pdf(qmia_binary_target, tmp_path):
    """Write_report=True should produce report.json and report.pdf."""
    out_dir = str(tmp_path / "qmia_pdf")
    attack_obj = QMIAAttack(output_dir=out_dir, write_report=True)

    output = attack_obj.attack(qmia_binary_target)

    assert output
    assert os.path.isfile(os.path.join(out_dir, "report.pdf"))
    assert os.path.isfile(os.path.join(out_dir, "report.json"))


def test_qmia_attackable_rejects_model_without_predict_proba():
    """Attackable() should reject a target whose model lacks predict_proba."""
    target = MagicMock(spec=Target)
    target.has_model.return_value = True
    target.has_data.return_value = True
    target.model = MagicMock(spec=[])  # no predict_proba
    assert not QMIAAttack.attackable(target)


def test_qmia_attack_signal_direction(qmia_binary_target, tmp_path):
    """AUC should exceed 0.5, confirming the attack distinguishes members."""
    attack_obj = QMIAAttack(
        output_dir=str(tmp_path / "qmia"),
        write_report=False,
    )

    output = attack_obj.attack(qmia_binary_target)
    instance = output["attack_experiment_logger"]["attack_instance_logger"][
        "instance_0"
    ]

    assert instance["AUC"] > 0.5


# ---------------------------------------------------------------------------
# Regression tests for C1 (degenerate regressor) and C2 (calibration tracking)
# ---------------------------------------------------------------------------


@pytest.fixture(name="qmia_degenerate_target")
def fixture_qmia_degenerate_target() -> Target:
    """Return a target whose hinge scores are identically zero.

    ``DummyClassifier(strategy="uniform")`` returns ``predict_proba=[0.5,0.5]``
    for every sample, so ``qmia_hinge_score`` collapses to zero and the
    quantile regressor degenerates to a constant predictor.
    """
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
    model: DummyClassifier = DummyClassifier(strategy="uniform", random_state=7)
    model.fit(X_train, y_train)
    target: Target = Target(
        model=model,
        dataset_name="qmia_dummy_uniform",
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


def test_qmia_raises_on_degenerate_regressor(
    qmia_degenerate_target: Target, tmp_path: Path
) -> None:
    """C1: QMIA must raise when the quantile regressor collapses to a constant."""
    attack_obj: QMIAAttack = QMIAAttack(
        output_dir=str(tmp_path / "qmia_degen"),
        write_report=False,
        alpha=0.01,
    )

    with pytest.raises(RuntimeError, match="degenerated to a near-constant"):
        attack_obj.attack(qmia_degenerate_target)


def test_qmia_metrics_include_calibration_ok(
    qmia_binary_target: Target, tmp_path: Path
) -> None:
    """C2: every QMIA metrics dict must expose a calibration_ok boolean flag."""
    attack_obj: QMIAAttack = QMIAAttack(
        output_dir=str(tmp_path / "qmia_calib"),
        write_report=False,
        alpha=0.2,
    )

    output: dict = attack_obj.attack(qmia_binary_target)
    m: dict = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]

    assert "calibration_ok" in m
    assert isinstance(m["calibration_ok"], bool)


def test_qmia_warns_on_miscalibration(
    qmia_binary_target: Target,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2: QMIA must warn and set calibration_ok=False when obs_fpr drifts.

    Forces adversarial thresholds (all zeros) so every sample with a positive
    hinge score is predicted member. On a healthy target this pushes
    observed_public_fpr far above alpha, exercising the C2 warning path.
    """

    def very_low_predict(_self, X: np.ndarray) -> np.ndarray:
        # Small variance to clear C1; all-negative so every positive score
        # crosses the threshold → obs_fpr ≈ 1.0, far from any realistic alpha.
        return np.linspace(-100.0, -99.0, len(X))

    monkeypatch.setattr(HistGradientBoostingRegressor, "predict", very_low_predict)
    caplog.set_level(logging.WARNING, logger="sacroml.attacks.qmia_attack")

    attack_obj: QMIAAttack = QMIAAttack(
        output_dir=str(tmp_path / "qmia_miscal"),
        write_report=False,
        alpha=0.01,
    )

    output: dict = attack_obj.attack(qmia_binary_target)
    m: dict = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]

    assert m["calibration_ok"] is False
    assert any("calibration deviated" in rec.message for rec in caplog.records)
