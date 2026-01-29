"""Test Scikit-learn with dataset class handling."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sacroml.attacks.attribute_attack import AttributeAttack
from sacroml.attacks.likelihood_attack import LIRAAttack
from sacroml.attacks.structural_attack import StructuralAttack
from sacroml.attacks.target import Target
from sacroml.attacks.worst_case_attack import WorstCaseAttack
from tests.attacks.sklearn_dataset import Nursery

output_dir = "output_sklearn"
target_dir = "target_sklearn"

random_state = 2


def test_sklearn() -> None:
    """Test scikit-learn with dataset class handling."""
    # Access data
    handler = Nursery()

    # Create data splits
    indices_train, indices_test = handler.get_train_test_indices()

    # Get data
    X, y = handler.get_data()
    X_train, y_train = handler.get_subset(X, y, indices_train)

    model = RandomForestClassifier(bootstrap=False)
    model.fit(X_train, y_train)

    target = Target(
        model=model,
        dataset_name="Nursery",
        dataset_module_path="tests/attacks/sklearn_dataset.py",
        indices_train=indices_train,
        indices_test=indices_test,
    )

    for i, index in enumerate(handler.feature_indices):
        target.add_feature(
            name=handler.feature_names[i],
            indices=index,
            encoding="onehot",
        )

    target.save(target_dir)

    target = Target()
    target.load(target_dir)

    assert target.X_train is not None
    assert target.y_train is not None
    assert target.X_test is not None
    assert target.y_test is not None

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
    output = attack.attack(target)
    assert output

    metrics = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]
    assert metrics["AUC"] > 0

    # Test structural attack
    attack = StructuralAttack(output_dir=output_dir)
    output = attack.attack(target)
    assert output

    # Test attribute attack
    attack = AttributeAttack(n_cpu=2, output_dir=output_dir)
    output = attack.attack(target)
    assert output

    # Test LiRA attack
    attack = LIRAAttack(n_shadow_models=100, output_dir=output_dir)
    output = attack.attack(target)
    assert output

    metrics = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]
    assert metrics["AUC"] > 0

    # Test generalisation function
    res = target.model.get_generalisation_error(
        target.X_train, target.y_train, target.X_test, target.y_test
    )
    assert res < 0

    # test get label indexes
    records = np.array(["a", "b", "c", "a", "c", "b"]).reshape(-1, 1)
    indices = np.array([0, 1, 2, 0, 2, 1])

    le = LabelEncoder().fit_transform(records)
    assert (target.model.get_label_indices(le) == indices).all()

    oh = OneHotEncoder().fit_transform(records)
    assert (target.model.get_label_indices(oh) == indices).all()

    oh1 = OneHotEncoder(sparse_output=False).fit_transform(records)
    assert (target.model.get_label_indices(oh1) == indices).all()

    # test get_losses
    testloss = target.model.get_losses(target.X_test, target.y_test)
    predictions = target.model.predict_proba(target.X_test)
    indices = target.model.get_label_indices(target.y_test)
    for i in range(len(target.y_test)):
        assert testloss[i] == 1.0 - predictions[i][indices[i]]
