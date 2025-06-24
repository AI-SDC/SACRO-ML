"""Test Pytorch model handling."""

from __future__ import annotations

import torch

from sacroml.attacks.attribute_attack import AttributeAttack
from sacroml.attacks.likelihood_attack import LIRAAttack
from sacroml.attacks.structural_attack import StructuralAttack
from sacroml.attacks.target import Target
from sacroml.attacks.worst_case_attack import WorstCaseAttack
from tests.attacks.pytorch_dataset import SyntheticData
from tests.attacks.pytorch_model import SimpleNet
from tests.attacks.pytorch_train import train

output_dir = "output_pytorch"
target_dir = "target_pytorch"

random_state = 2
torch.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)


def test_pytorch() -> None:  # pylint:disable=too-many-locals
    """Test pytorch handling."""
    # Make some data
    x_dim = 4
    y_dim = 4
    dataset = SyntheticData(x_dim=x_dim, y_dim=y_dim, random_state=random_state)
    train_dataloader = dataset.get_train_loader()
    _ = dataset.get_test_loader()

    model_params = {
        "x_dim": x_dim,
        "y_dim": y_dim,
    }
    train_params = {
        "epochs": 10,
        "learning_rate": 0.001,
        "momentum": 0.9,
    }

    # Make and fit pytorch model
    model = SimpleNet(**model_params)
    train(model, train_dataloader, **train_params)

    # Create Target wrapper
    target = Target(
        model=model,
        model_module_path="tests/attacks/pytorch_model.py",
        model_params=model_params,
        train_module_path="tests/attacks/pytorch_train.py",
        train_params=train_params,
        dataset_name="synthetic",
        # processed data
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_test=dataset.X_test,
        y_test=dataset.y_test,
        # original unprocessed data
        X_train_orig=dataset.X_train,
        y_train_orig=dataset.y_train,
        X_test_orig=dataset.X_test,
        y_test_orig=dataset.y_test,
    )
    # Add feature details for attribute attack
    for i in range(dataset.X_train.shape[1]):
        target.add_feature(
            name=f"A{i}",
            indices=[i],
            encoding="float",
        )

    # Test saving and loading
    target.save(target_dir)

    loaded_target = Target()
    loaded_target.load(target_dir)
    assert loaded_target.dataset_name == target.dataset_name

    score = target.model.score(dataset.X_test, dataset.y_test)
    loaded_score = loaded_target.model.score(dataset.X_test, dataset.y_test)
    assert score == loaded_score

    # Test worst case attack
    attack_obj = WorstCaseAttack(
        n_reps=10,
        n_dummy_reps=1,
        train_beta=5,
        test_beta=2,
        p_thresh=0.05,
        test_prop=0.5,
        output_dir=output_dir,
    )
    output = attack_obj.attack(target)
    assert output

    metrics = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]
    assert metrics["AUC"] > 0

    # Test structural attack
    obj = StructuralAttack(output_dir=output_dir)
    output = obj.attack(target)
    assert not output  # expected not to run

    # Test attribute attack
    obj = AttributeAttack(n_cpu=2, output_dir=output_dir)
    output = obj.attack(target)
    assert output

    # Test LiRA attack
    obj = LIRAAttack(n_shadow_models=100, output_dir=output_dir)
    output = obj.attack(target)
    assert output

    metrics = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]
    assert metrics["AUC"] > 0
