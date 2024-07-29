"""Test attack factory."""

from __future__ import annotations

import json
import os

import pytest
import yaml
from sklearn.ensemble import RandomForestClassifier

from sacroml.attacks.factory import run_attacks
from sacroml.config.attack import _get_attack


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
    assert metrics["TPR"] == pytest.approx(0.89, abs=0.01)
    assert metrics["FPR"] == pytest.approx(0.43, abs=0.01)
