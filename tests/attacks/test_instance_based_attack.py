"""Tests for InstanceBasedAttack."""

import logging
import os

import numpy as np
import pytest
from sklearn.datasets import make_moons, make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, NuSVC
from sklearn.tree import DecisionTreeClassifier

from sacroml.attacks.factory import create_attack
from sacroml.attacks.instance_based_attack import (
    INSTANCE_MATCH_ATOL,
    InstanceBasedAttack,
)
from sacroml.attacks.target import Target
from sacroml.safemodel.classifiers.dp_svc import DPSVC


def _make_target_clf(model, n_samples=100, random_state=42):
    """Create a target with a fitted classification model on synthetic data."""
    X, y = make_moons(n_samples=n_samples, noise=0.3, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    model.fit(X_train, y_train)
    return Target(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def _make_target_reg(model, n_samples=100, random_state=42):
    """Create a target with a fitted regression model on synthetic data."""
    X, y = make_regression(
        n_samples=n_samples, n_features=2, noise=0.1, random_state=random_state
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    model.fit(X_train, y_train)
    return Target(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


class TestAttackable:
    """Tests for the attackable classmethod."""

    def test_no_model(self):
        """Test attackable returns False with no model."""
        target = Target()
        assert not InstanceBasedAttack.attackable(target)

    def test_no_data(self):
        """Test attackable returns False with no training data."""
        model = SVC(gamma=0.1)
        X, y = make_moons(n_samples=50, noise=0.3, random_state=42)
        model.fit(X, y)
        target = Target(model=model)
        assert not InstanceBasedAttack.attackable(target)

    def test_valid_target(self):
        """Test attackable returns True for valid target."""
        target = _make_target_clf(SVC(gamma=0.1))
        assert InstanceBasedAttack.attackable(target)

    def test_non_instance_model_attackable(self):
        """Test attackable returns True for non-instance models too."""
        target = _make_target_clf(DecisionTreeClassifier(random_state=42))
        assert InstanceBasedAttack.attackable(target)


class TestSVMDetection:
    """Tests for SVM model detection and leakage confirmation."""

    def test_svc_detects_leakage(self):
        """Test SVC support vector leakage is detected."""
        target = _make_target_clf(SVC(gamma=0.1))
        attack = InstanceBasedAttack(
            output_dir="outputs_instance_based", write_report=False
        )
        output = attack.attack(target)

        assert output
        instance = output["attack_experiment_logger"]["attack_instance_logger"][
            "instance_0"
        ]
        assert instance["is_instance_based"] is True
        assert instance["model_type"] == "SVC"
        assert instance["data_leakage_confirmed"] is True
        assert instance["n_matched"] > 0
        assert instance["match_fraction"] > 0
        assert instance["n_stored_instances"] > 0
        assert len(instance["mitigations"]) > 0

    def test_nusvc_detects_leakage(self):
        """Test NuSVC support vector leakage is detected."""
        target = _make_target_clf(NuSVC())
        attack = InstanceBasedAttack(
            output_dir="outputs_instance_based", write_report=False
        )
        output = attack.attack(target)

        instance = output["attack_experiment_logger"]["attack_instance_logger"][
            "instance_0"
        ]
        assert instance["is_instance_based"] is True
        assert instance["model_type"] == "NuSVC"
        assert instance["data_leakage_confirmed"] is True

    def test_svr_detects_leakage(self):
        """Test SVR support vector leakage is detected."""
        target = _make_target_reg(SVR())
        attack = InstanceBasedAttack(
            output_dir="outputs_instance_based", write_report=False
        )
        output = attack.attack(target)

        instance = output["attack_experiment_logger"]["attack_instance_logger"][
            "instance_0"
        ]
        assert instance["is_instance_based"] is True
        assert instance["model_type"] == "SVR"
        assert instance["data_leakage_confirmed"] is True


class TestKNNDetection:
    """Tests for kNN model detection and leakage confirmation."""

    def test_knn_detects_leakage(self):
        """Test KNeighborsClassifier stores all training data."""
        target = _make_target_clf(KNeighborsClassifier(n_neighbors=3))
        attack = InstanceBasedAttack(
            output_dir="outputs_instance_based", write_report=False
        )
        output = attack.attack(target)

        instance = output["attack_experiment_logger"]["attack_instance_logger"][
            "instance_0"
        ]
        assert instance["is_instance_based"] is True
        assert instance["model_type"] == "KNeighborsClassifier"
        assert instance["data_leakage_confirmed"] is True
        assert instance["storage_fraction"] == pytest.approx(1.0)
        assert instance["match_fraction"] == pytest.approx(1.0)

    def test_knn_regressor_leakage(self):
        """Test KNeighborsRegressor stores all training data."""
        target = _make_target_reg(KNeighborsRegressor(n_neighbors=3))
        attack = InstanceBasedAttack(
            output_dir="outputs_instance_based", write_report=False
        )
        output = attack.attack(target)

        instance = output["attack_experiment_logger"]["attack_instance_logger"][
            "instance_0"
        ]
        assert instance["is_instance_based"] is True
        assert instance["model_type"] == "KNeighborsRegressor"
        assert instance["data_leakage_confirmed"] is True
        assert instance["storage_fraction"] == pytest.approx(1.0)


class TestNonInstanceModels:
    """Tests for non-instance-based models."""

    def test_decision_tree_safe(self):
        """Test DecisionTree is not flagged as instance-based."""
        target = _make_target_clf(DecisionTreeClassifier(random_state=42))
        attack = InstanceBasedAttack(
            output_dir="outputs_instance_based", write_report=False
        )
        output = attack.attack(target)

        instance = output["attack_experiment_logger"]["attack_instance_logger"][
            "instance_0"
        ]
        assert instance["is_instance_based"] is False
        assert instance["data_leakage_confirmed"] is False
        assert instance["n_stored_instances"] == 0
        assert instance["n_matched"] == 0


class TestDPSVC:
    """Tests for differentially private SVM variant."""

    def test_dpsvc_is_dp_safe(self):
        """Test DPSVC is detected as DP-safe."""
        X, y = make_moons(n_samples=100, noise=0.3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = DPSVC(eps=10)
        model.fit(X_train, y_train)
        target = Target(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        attack = InstanceBasedAttack(
            output_dir="outputs_instance_based", write_report=False
        )
        output = attack.attack(target)

        instance = output["attack_experiment_logger"]["attack_instance_logger"][
            "instance_0"
        ]
        assert instance["is_dp_safe"] is True
        assert any("DP-safe" in m for m in instance["mitigations"])


class TestPipeline:
    """Tests for Pipeline unwrapping."""

    def test_pipeline_svc(self):
        """Test SVC inside a Pipeline is detected and leakage confirmed."""
        model = Pipeline([("scaler", StandardScaler()), ("svc", SVC(gamma=0.1))])
        X, y = make_moons(n_samples=100, noise=0.3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        target = Target(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        attack = InstanceBasedAttack(
            output_dir="outputs_instance_based", write_report=False
        )
        output = attack.attack(target)

        instance = output["attack_experiment_logger"]["attack_instance_logger"][
            "instance_0"
        ]
        assert instance["is_instance_based"] is True
        assert instance["model_type"] == "SVC"
        assert instance["data_leakage_confirmed"] is True
        assert instance["n_matched"] > 0


class TestConfiguration:
    """Tests for attack configuration and parameters."""

    def test_n_examples_does_not_cap_recorded_matches(self):
        """n_examples limits report display only, not recorded matches."""
        target = _make_target_clf(SVC(gamma=0.1))
        attack = InstanceBasedAttack(
            output_dir="outputs_instance_based",
            write_report=False,
            n_examples=2,
            report_individual=True,
        )
        output = attack.attack(target)

        instance = output["attack_experiment_logger"]["attack_instance_logger"][
            "instance_0"
        ]
        individual = instance["individual"]
        # Every stored instance is recorded despite the small n_examples.
        assert len(individual["matched"]) == instance["n_stored_instances"]
        assert instance["n_stored_instances"] > attack.n_examples
        assert sum(individual["matched"]) == instance["n_matched"]

    def test_str_representation(self):
        """Test __str__ returns the attack name."""
        attack = InstanceBasedAttack(
            output_dir="outputs_instance_based", write_report=False
        )
        assert str(attack) == "Instance-Based Model Attack"

    def test_get_params(self):
        """Test get_params returns constructor parameters."""
        attack = InstanceBasedAttack(
            output_dir="outputs_instance_based",
            write_report=False,
            n_examples=5,
            atol=1e-6,
        )
        params = attack.get_params()
        assert params["n_examples"] == 5
        assert params["atol"] == 1e-6
        assert params["output_dir"] == "outputs_instance_based"
        assert params["report_individual"] is False

    def test_default_atol_is_module_constant(self):
        """Default atol matches INSTANCE_MATCH_ATOL, see issue #454."""
        attack = InstanceBasedAttack(write_report=False)
        assert attack.atol == INSTANCE_MATCH_ATOL
        assert INSTANCE_MATCH_ATOL == 1e-8

    def test_factory_registration(self):
        """Test attack is registered in the factory."""
        attack = create_attack(
            "instance_based",
            output_dir="outputs_instance_based",
            write_report=False,
        )
        assert isinstance(attack, InstanceBasedAttack)


class TestOutputStructure:
    """Tests for output format and report generation."""

    def test_output_structure(self):
        """Test output dict has required keys and metadata."""
        target = _make_target_clf(SVC(gamma=0.1))
        attack = InstanceBasedAttack(
            output_dir="outputs_instance_based", write_report=False
        )
        output = attack.attack(target)

        assert "log_id" in output
        assert "log_time" in output
        assert "metadata" in output
        assert "attack_experiment_logger" in output

        metadata = output["metadata"]
        assert metadata["attack_name"] == "Instance-Based Model Attack"
        assert "global_metrics" in metadata
        assert "data_leakage_confirmed" in metadata["global_metrics"]

    def test_report_files_created(self):
        """Test JSON and PDF report files are generated."""
        target = _make_target_clf(SVC(gamma=0.1))
        attack = InstanceBasedAttack(
            output_dir="outputs_instance_based", write_report=True
        )
        attack.attack(target)

        assert os.path.exists(os.path.join("outputs_instance_based", "report.json"))
        assert os.path.exists(os.path.join("outputs_instance_based", "report.pdf"))

    def test_record_level_individual_structure(self):
        """Per-record individual block has parallel lists of equal length."""
        target = _make_target_clf(SVC(gamma=0.1))
        attack = InstanceBasedAttack(
            output_dir="outputs_instance_based",
            write_report=False,
            report_individual=True,
        )
        output = attack.attack(target)

        instance = output["attack_experiment_logger"]["attack_instance_logger"][
            "instance_0"
        ]
        individual = instance["individual"]
        for key in ("stored_index", "training_index", "matched", "stored_values"):
            assert key in individual

        n_stored = instance["n_stored_instances"]
        assert len(individual["stored_index"]) == n_stored
        assert len(individual["training_index"]) == n_stored
        assert len(individual["matched"]) == n_stored
        assert len(individual["stored_values"]) == n_stored
        assert isinstance(individual["stored_values"][0], list)
        # A non-negative training index appears exactly when matched is True.
        for train_idx, is_match in zip(
            individual["training_index"], individual["matched"], strict=True
        ):
            assert (train_idx >= 0) == is_match

    def test_report_individual_off_by_default(self):
        """Without report_individual, no per-record block is emitted."""
        target = _make_target_clf(SVC(gamma=0.1))
        attack = InstanceBasedAttack(
            output_dir="outputs_instance_based", write_report=False
        )
        output = attack.attack(target)

        instance = output["attack_experiment_logger"]["attack_instance_logger"][
            "instance_0"
        ]
        assert "individual" not in instance
        assert "example_matches" not in instance

    def test_empty_target_returns_empty(self):
        """Test attack on empty target returns empty dict."""
        target = Target()
        attack = InstanceBasedAttack(
            output_dir="outputs_instance_based", write_report=False
        )
        output = attack.attack(target)
        assert output == {}


class TestGracefulDegradation:
    """Tests for defensive / graceful-degradation paths."""

    def test_unfitted_knn_warns_and_returns_zero(self, caplog):
        """Unfitted kNN: no _fit_X attribute triggers a warning and zero stored."""
        X_train = np.array([[0.0, 0.0], [1.0, 1.0]])
        y_train = np.array([0, 1])
        unfitted = KNeighborsClassifier(n_neighbors=1)
        target = Target(
            model=unfitted,
            X_train=X_train,
            y_train=y_train,
            X_test=X_train,
            y_test=y_train,
        )
        attack = InstanceBasedAttack(
            output_dir="outputs_instance_based", write_report=False
        )
        caplog.set_level(
            logging.WARNING, logger="sacroml.attacks.instance_based_attack"
        )
        output = attack.attack(target)

        instance = output["attack_experiment_logger"]["attack_instance_logger"][
            "instance_0"
        ]
        assert instance["model_type"] == "KNeighborsClassifier"
        assert instance["n_stored_instances"] == 0
        assert instance["data_leakage_confirmed"] is False
        assert any("_fit_X" in rec.message for rec in caplog.records)

    def test_feature_dim_mismatch_warns_and_returns(self, caplog):
        """Stored support vectors and X_train with different feature counts."""
        X_train = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        y_train = np.array([0, 1, 0, 1])
        model = SVC(gamma=0.1).fit(X_train, y_train)
        target = Target(
            model=model,
            X_train=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            y_train=np.array([0, 1]),
            X_test=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            y_test=np.array([0, 1]),
        )
        attack = InstanceBasedAttack(
            output_dir="outputs_instance_based", write_report=False
        )
        caplog.set_level(
            logging.WARNING, logger="sacroml.attacks.instance_based_attack"
        )
        output = attack.attack(target)

        instance = output["attack_experiment_logger"]["attack_instance_logger"][
            "instance_0"
        ]
        assert instance["n_matched"] == 0
        assert instance["data_leakage_confirmed"] is False
        assert instance["details"]["error"] == "Feature dimension mismatch"
        assert any(
            "Feature dimension mismatch" in rec.message for rec in caplog.records
        )

    def test_fallback_search_when_indices_wrong(self):
        """Wrong support_ index hint forces the fallback brute-force loop."""
        X_train = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        y_train = np.array([0, 1, 0, 1])
        model = SVC(gamma=0.1).fit(X_train, y_train)
        model.support_ = np.zeros_like(model.support_)
        target = Target(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_train,
            y_test=y_train,
        )
        attack = InstanceBasedAttack(
            output_dir="outputs_instance_based", write_report=False
        )
        output = attack.attack(target)

        instance = output["attack_experiment_logger"]["attack_instance_logger"][
            "instance_0"
        ]
        assert instance["data_leakage_confirmed"] is True
        assert instance["n_matched"] > 0
