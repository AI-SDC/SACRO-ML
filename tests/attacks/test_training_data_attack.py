"""Test training data in model attack."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sacroml.attacks.target import Target
from sacroml.attacks.training_data_attack import (
    TrainingDataInModelAttack,
    _find_matches,
    _get_final_estimator,
    _get_stored_vectors_and_train_data,
    _has_dp_embedding,
    _is_attackable_model,
)


def get_target_svc(**kwargs) -> Target:
    """Create target with SVC model."""
    X, y = make_moons(n_samples=50, noise=0.5, random_state=12345)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = SVC(probability=True, gamma=0.1, **kwargs)
    model.fit(X_train, y_train)
    return Target(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def get_target_knn(**kwargs) -> Target:
    """Create target with KNeighborsClassifier model."""
    X, y = make_moons(n_samples=50, noise=0.5, random_state=12345)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = KNeighborsClassifier(**kwargs)
    model.fit(X_train, y_train)
    return Target(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def get_target_dt() -> Target:
    """Create target with DecisionTree model (not attackable)."""
    X, y = make_moons(n_samples=50, noise=0.5, random_state=12345)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return Target(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def get_target_pipeline_svc() -> Target:
    """Create target with Pipeline(StandardScaler, SVC)."""
    X, y = make_moons(n_samples=50, noise=0.5, random_state=12345)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(probability=True, gamma=0.1)),
        ]
    )
    model.fit(X_train, y_train)
    return Target(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def test_attackable_svc():
    """SVC target should be attackable."""
    target = get_target_svc()
    assert TrainingDataInModelAttack.attackable(target)


def test_attackable_knn():
    """KNeighborsClassifier target should be attackable."""
    target = get_target_knn()
    assert TrainingDataInModelAttack.attackable(target)


def test_attackable_decision_tree():
    """DecisionTree target should not be attackable."""
    target = get_target_dt()
    assert not TrainingDataInModelAttack.attackable(target)


def test_attackable_pipeline_svc():
    """Pipeline with SVC should be attackable."""
    target = get_target_pipeline_svc()
    assert TrainingDataInModelAttack.attackable(target)


def test_attackable_no_model():
    """Target without model should not be attackable."""
    X, y = make_moons(n_samples=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    target = Target(
        model=None,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert not TrainingDataInModelAttack.attackable(target)


def test_attack_svc_finds_matches():
    """SVC attack should find matches between support vectors and training data."""
    target = get_target_svc()
    attack = TrainingDataInModelAttack(write_report=False)
    output = attack.attack(target)

    assert output
    assert "attack_experiment_logger" in output
    instances = output["attack_experiment_logger"]["attack_instance_logger"]
    metrics = instances["instance_0"]

    assert metrics["model_type"] == "SVC"
    assert metrics["contains_training_data"] is True
    assert metrics["n_stored"] > 0
    assert metrics["n_matches"] > 0
    assert len(metrics["example_matches"]) <= 10
    assert "mitigations" in metrics


def test_attack_knn_reports_full_data():
    """KNN attack should report that full training data is stored."""
    target = get_target_knn()
    attack = TrainingDataInModelAttack(write_report=False)
    output = attack.attack(target)

    assert output
    instances = output["attack_experiment_logger"]["attack_instance_logger"]
    metrics = instances["instance_0"]

    assert metrics["model_type"] == "KNeighborsClassifier"
    assert metrics["contains_training_data"] is True
    assert metrics["n_stored"] == target.X_train.shape[0]
    assert metrics["n_matches"] == target.X_train.shape[0]


def test_attack_pipeline_svc():
    """Pipeline with SVC should find matches in transformed space."""
    target = get_target_pipeline_svc()
    attack = TrainingDataInModelAttack(write_report=False)
    output = attack.attack(target)

    assert output
    instances = output["attack_experiment_logger"]["attack_instance_logger"]
    metrics = instances["instance_0"]

    assert metrics["model_type"] == "SVC"
    assert metrics["contains_training_data"] is True
    assert metrics["n_matches"] > 0


def test_get_final_estimator_standalone():
    """_get_final_estimator returns model for non-Pipeline."""
    model = SVC()
    assert _get_final_estimator(model) is model


def test_get_final_estimator_pipeline():
    """_get_final_estimator returns last step for Pipeline."""
    model = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
    final = _get_final_estimator(model)
    assert isinstance(final, SVC)


def test_has_dp_embedding_false():
    """Regular SVC should not have DP embedding."""
    model = SVC()
    assert not _has_dp_embedding(model)


def test_is_attackable_model_svc():
    """SVC should be attackable."""
    assert _is_attackable_model(SVC())


def test_is_attackable_model_knn():
    """KNeighborsClassifier should be attackable."""
    assert _is_attackable_model(KNeighborsClassifier())


def test_is_attackable_model_dt():
    """DecisionTree should not be attackable."""
    assert not _is_attackable_model(DecisionTreeClassifier())


def test_find_matches():
    """_find_matches should identify matching rows."""
    stored = np.array([[1.0, 2.0], [3.0, 4.0]])
    train_data = np.array([[1.0, 2.0], [5.0, 6.0], [3.0, 4.0]])
    matches = _find_matches(stored, train_data)
    assert len(matches) == 2
    assert matches[0]["train_idx"] == 0
    assert matches[0]["stored_idx"] == 0
    assert matches[1]["train_idx"] == 2
    assert matches[1]["stored_idx"] == 1


def test_get_stored_vectors_svc():
    """_get_stored_vectors_and_train_data for SVC."""
    target = get_target_svc()
    model = target.model.model
    stored, train_data, model_type = _get_stored_vectors_and_train_data(
        model, target.X_train
    )
    assert model_type == "SVC"
    assert stored.shape[0] > 0
    assert stored.shape[1] == train_data.shape[1]


def test_get_stored_vectors_knn():
    """_get_stored_vectors_and_train_data for KNN."""
    target = get_target_knn()
    model = target.model.model
    stored, train_data, model_type = _get_stored_vectors_and_train_data(
        model, target.X_train
    )
    assert model_type == "KNeighborsClassifier"
    assert stored.shape == train_data.shape


def test_attack_str():
    """Attack string representation."""
    attack = TrainingDataInModelAttack()
    assert "Training Data" in str(attack)


def test_attackable_svc_precomputed_kernel():
    """SVC with kernel='precomputed' should not be attackable."""
    X, y = make_moons(n_samples=30, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    from sklearn.metrics.pairwise import rbf_kernel

    gram_train = rbf_kernel(X_train, X_train)
    model = SVC(kernel="precomputed")
    model.fit(gram_train, y_train)
    target = Target(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert not TrainingDataInModelAttack.attackable(target)
