"""Test attack factory."""

from __future__ import annotations

import json
import os

import pytest
import yaml
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sacroml.attacks.factory import create_attack, run_attacks
from sacroml.attacks.qmia_attack import QMIAAttack
from sacroml.attacks.target import Target
from sacroml.config.attack import _get_attack


def _make_binary_target() -> Target:
    """Return a small binary target for QMIA factory tests."""
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        class_sep=1.2,
        random_state=11,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=11
    )

    model = RandomForestClassifier(n_estimators=40, random_state=11)
    model.fit(X_train, y_train)
    return Target(
        model=model,
        dataset_name="qmia_factory_binary",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_train_orig=X_train,
        y_train_orig=y_train,
        X_test_orig=X_test,
        y_test_orig=y_test,
    )


@pytest.mark.parametrize(
    "get_target", [RandomForestClassifier(random_state=1)], indirect=True
)
def test_factory(monkeypatch, get_target):
    """Test Target object creation, saving, and loading."""
    # create target_dir
    target = get_target
    target.save("target_factory")

    model = target.model
    assert model.score(target.X_test, target.y_test) == pytest.approx(0.92, 0.01)

    # create LiRA config with default params
    mock_input = "yes"
    monkeypatch.setattr("builtins.input", lambda _: mock_input)
    attacks = [_get_attack("lira")]
    attacks[0]["params"]["output_dir"] = "outputs_factory"

    # create attack.yaml
    filename: str = "attack.yaml"
    with open(filename, "w", encoding="utf-8") as fp:
        yaml.dump({"attacks": attacks}, fp)

    # run attacks
    run_attacks("target_factory", "attack.yaml")

    # load JSON report
    path = os.path.normpath("outputs_factory/report.json")
    with open(path, encoding="utf-8") as fp:
        report = json.load(fp)

    # check report output
    nr = list(report.keys())[0]
    metrics = report[nr]["attack_experiment_logger"]["attack_instance_logger"][
        "instance_0"
    ]
    assert metrics["TPR"] == pytest.approx(0.91, abs=0.01)
    assert metrics["FPR"] == pytest.approx(0.41, abs=0.01)


def test_factory_qmia(monkeypatch, tmp_path):
    """Test attack factory wiring for QMIA."""
    attack_obj = create_attack("qmia")
    assert isinstance(attack_obj, QMIAAttack)

    target = _make_binary_target()
    target_dir = tmp_path / "target_factory_qmia"
    output_dir = tmp_path / "outputs_factory_qmia"
    attack_filename = tmp_path / "attack_qmia.yaml"
    target.save(str(target_dir))

    mock_input = "yes"
    monkeypatch.setattr("builtins.input", lambda _: mock_input)
    attacks = [_get_attack("qmia")]
    attacks[0]["params"]["output_dir"] = str(output_dir)

    with open(attack_filename, "w", encoding="utf-8") as fp:
        yaml.dump({"attacks": attacks}, fp)

    run_attacks(str(target_dir), str(attack_filename))

    path = os.path.normpath(f"{output_dir}/report.json")
    with open(path, encoding="utf-8") as fp:
        report = json.load(fp)

    nr = list(report.keys())[0]
    metrics = report[nr]["attack_experiment_logger"]["attack_instance_logger"][
        "instance_0"
    ]
    assert 0 <= metrics["TPR"] <= 1
    assert 0 <= metrics["FPR"] <= 1
