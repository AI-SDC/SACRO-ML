"""Test Pytorch model handling."""

from __future__ import annotations

import torch

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
    assert not output  # expected not to run

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
