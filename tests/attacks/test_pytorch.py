"""Test Pytorch model handling."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim

from sacroml.attacks import (
    attribute_attack,
    likelihood_attack,
    structural_attack,
    worst_case_attack,
)
from sacroml.attacks.target import Target

output_dir = "output_pytorch"
target_dir = "target_pytorch"

random_state = 2
torch.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)


class SimpleNet(nn.Module):
    """A simple Pytorch classification model."""

    def __init__(self, x_dim: int, y_dim: int) -> None:
        """Construct a simple Pytorch model."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(x_dim, 50),
            nn.ReLU(),
            nn.Linear(50, y_dim),
        )
        self.epochs = 100
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.layers.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagate input."""
        return self.layers(x)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit model to data."""
        x_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)

        for _ in range(self.epochs):
            # Forward
            logits = self(x_train_tensor)
            loss = self.criterion(logits, y_train_tensor)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def test_pytorch() -> None:  # pylint:disable=too-many-locals
    """Test pytorch handling."""
    # Make some data
    n_features = 4
    n_classes = 4
    X_orig, y_orig = make_classification(
        n_samples=50,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    X_orig = np.asarray(X_orig)
    y_orig = np.asarray(y_orig)

    # Preprocess
    input_encoder = StandardScaler()
    X = input_encoder.fit_transform(X_orig)
    y = y_orig  # leave as labels

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, shuffle=True, random_state=random_state
    )

    X = np.asarray(X)
    y = np.asarray(y)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    # Make and fit pytorch model
    model = SimpleNet(x_dim=n_features, y_dim=n_classes)
    model.fit(X_train, y_train)

    # Create Target wrapper
    target = Target(
        model=model,
        dataset_name="synthetic",
        # processed data
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        # original unprocessed data
        X_orig=X_orig,
        y_orig=y_orig,
        X_train_orig=X_train,
        y_train_orig=y_train,
        X_test_orig=X_test,
        y_test_orig=y_test,
    )
    # Add feature details for attribute attack
    for i in range(X.shape[1]):
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

    score = target.model.score(X_test, y_test)
    loaded_score = loaded_target.model.score(X_test, y_test)
    assert score == loaded_score

    # Test worst case attack
    attack_obj = worst_case_attack.WorstCaseAttack(
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
    assert metrics["AUC"] >= 0.5

    # Test structural attack
    obj = structural_attack.StructuralAttack(output_dir=output_dir)
    output = obj.attack(target)
    assert not output  # expected not to run

    # Test attribute attack
    obj = attribute_attack.AttributeAttack(n_cpu=2, output_dir=output_dir)
    output = obj.attack(target)
    assert output

    # Test LiRA attack
    obj = likelihood_attack.LIRAAttack(n_shadow_models=100, output_dir=output_dir)
    output = obj.attack(target)
    assert output

    metrics = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]
    assert metrics["AUC"] >= 0.5
