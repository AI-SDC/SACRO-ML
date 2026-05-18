"""Test Pytorch model handling."""

from __future__ import annotations

import logging

import numpy as np
import torch
import yaml

from sacroml.attacks.attribute_attack import AttributeAttack
from sacroml.attacks.likelihood_attack import LIRAAttack
from sacroml.attacks.structural_attack import StructuralAttack
from sacroml.attacks.target import Target
from sacroml.attacks.worst_case_attack import WorstCaseAttack
from tests.attacks.pytorch_dataset import Synthetic
from tests.attacks.pytorch_model import OverfitNet
from tests.attacks.pytorch_train import train

output_dir = "output_pytorch"
target_dir = "target_pytorch"

random_state = 2
torch.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)


def test_pytorch() -> None:
    """Test PyTorch handling."""
    # Access dataset
    handler = Synthetic()

    # Get the (preprocessed) dataset
    dataset = handler.get_dataset()

    # Create data splits
    indices_train, indices_test = handler.get_train_test_indices()

    # Get dataloaders
    train_loader = handler.get_dataloader(dataset, indices_train)

    # Make and fit model
    model_params = {
        "x_dim": 4,
        "y_dim": 4,
        "n_units": 1000,
    }
    train_params = {
        "epochs": 10,
        "learning_rate": 0.001,
        "momentum": 0.9,
    }
    model = OverfitNet(**model_params)

    train(model, train_loader, **train_params)

    # Wrap model and data
    target = Target(
        model=model,
        model_module_path="tests/attacks/pytorch_model.py",
        model_params=model_params,
        train_module_path="tests/attacks/pytorch_train.py",
        train_params=train_params,
        dataset_module_path="tests/attacks/pytorch_dataset.py",
        dataset_name="Synthetic",
        indices_train=indices_train,
        indices_test=indices_test,
    )

    for i in range(model_params["x_dim"]):
        target.add_feature(
            name=f"X{i}",
            indices=[i],
            encoding="float",
        )

    # Test saving and loading
    target.save(target_dir)

    tgt = Target()
    tgt.load(target_dir)

    assert tgt.dataset_name == target.dataset_name
    assert tgt.X_train is not None
    assert tgt.y_train is not None
    assert tgt.X_test is not None
    assert tgt.y_test is not None

    # Test worst case attack
    attack = WorstCaseAttack(
        n_reps=10,
        n_dummy_reps=1,
        train_beta=5,
        test_beta=2,
        p_thresh=0.05,
        test_prop=0.5,
        output_dir=output_dir,
    )
    output = attack.attack(tgt)
    assert output

    metrics = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]
    assert metrics["AUC"] > 0

    # Test structural attack
    attack = StructuralAttack(output_dir=output_dir)
    output = attack.attack(tgt)
    assert output  # expected to run

    # Test attribute attack
    attack = AttributeAttack(n_cpu=2, output_dir=output_dir)
    output = attack.attack(tgt)
    assert output

    # Test LiRA attack
    attack = LIRAAttack(n_shadow_models=100, output_dir=output_dir)
    output = attack.attack(tgt)
    assert output

    metrics = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]
    assert metrics["AUC"] > 0

    # Test generalisation function
    res = tgt.model.get_generalisation_error(
        tgt.X_train, tgt.y_train, tgt.X_test, tgt.y_test
    )
    assert res < 0

    # Test score function
    res = tgt.model.score(tgt.X_test, tgt.y_test)
    assert res > 0

    # Test predict function
    res = tgt.model.predict(tgt.X_test)
    assert len(res) > 0

    # test get_losses
    # test get_losses
    testloss = tgt.model.get_losses(tgt.X_test, tgt.y_test)
    predictions = tgt.model.predict_proba(tgt.X_test)
    indices = tgt.model.get_label_indices(tgt.y_test)
    for i in range(len(tgt.y_test)):
        assert testloss[i] == 1.0 - predictions[i][indices[i]]


def _make_target(batch_size: int | None) -> Target:
    """Build a small wrapped PyTorch Target for batch_size tests."""
    handler = Synthetic()
    dataset = handler.get_dataset()
    indices_train, indices_test = handler.get_train_test_indices()
    train_loader = handler.get_dataloader(dataset, indices_train)

    model_params = {"x_dim": 4, "y_dim": 4, "n_units": 1000}
    train_params = {"epochs": 1, "learning_rate": 0.001, "momentum": 0.9}
    model = OverfitNet(**model_params)
    train(model, train_loader, **train_params)

    return Target(
        model=model,
        model_module_path="tests/attacks/pytorch_model.py",
        model_params=model_params,
        train_module_path="tests/attacks/pytorch_train.py",
        train_params=train_params,
        batch_size=batch_size,
        dataset_module_path="tests/attacks/pytorch_dataset.py",
        dataset_name="Synthetic",
        indices_train=indices_train,
        indices_test=indices_test,
    )


def test_batch_size_roundtrip() -> None:
    """An explicit batch_size survives Target.save() / load()."""
    target = _make_target(batch_size=64)
    assert target.model.batch_size == 64

    target_path = "target_pytorch_bs"
    target.save(target_path)

    # Top-level scalar in target.yaml, not nested inside train_params.
    with open(f"{target_path}/target.yaml", encoding="utf-8") as f:
        saved = yaml.safe_load(f)
    assert saved["batch_size"] == 64
    assert "batch_size" not in saved.get("train_params", {})

    tgt = Target()
    tgt.load(target_path)
    assert tgt.batch_size == 64
    assert tgt.model.batch_size == 64


def test_batch_size_defaults_to_32_when_unset() -> None:
    """An unset batch_size resolves to 32 on the wrapped model (no warning)."""
    target = _make_target(batch_size=None)
    assert target.model.batch_size == 32

    target_path = "target_pytorch_bs_none"
    target.save(target_path)
    # A freshly-saved target records the resolved value (32), not None.
    with open(f"{target_path}/target.yaml", encoding="utf-8") as f:
        saved = yaml.safe_load(f)
    assert saved["batch_size"] == 32


def test_batch_size_backcompat_old_yaml(caplog) -> None:
    """A hand-crafted old target.yaml lacking batch_size loads as 32 and warns.

    This deliberately writes the yaml by hand (rather than saving a Target
    with batch_size=None) because the absent-key path is distinct from the
    resolved-to-32 save path.
    """
    target = _make_target(batch_size=128)
    target_path = "target_pytorch_old"
    target.save(target_path)

    yaml_path = f"{target_path}/target.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        saved = yaml.safe_load(f)
    del saved["batch_size"]
    assert "batch_size" not in saved
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(saved, f, default_flow_style=False, sort_keys=False)

    tgt = Target()
    with caplog.at_level(logging.WARNING, logger="sacroml.attacks.target"):
        tgt.load(target_path)

    assert tgt.model.batch_size == 32
    warnings = [
        r
        for r in caplog.records
        if "no recorded batch_size" in r.message and r.levelno == logging.WARNING
    ]
    assert len(warnings) == 1


def test_fit_batch_size_override_warns(caplog) -> None:
    """Fit() warns when an explicit batch_size differs from the recorded one."""
    target = _make_target(batch_size=16)
    handler = Synthetic()
    X = np.asarray(handler.X, dtype=np.float64)
    y = np.asarray(handler.y, dtype=np.int64)

    with caplog.at_level(logging.WARNING, logger="sacroml.attacks.model_pytorch"):
        target.model.fit(X, y, batch_size=8)

    warnings = [r for r in caplog.records if "differs from the target" in r.message]
    assert len(warnings) == 1
