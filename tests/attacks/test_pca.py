"""Test scikit-learn PCA handling."""

from __future__ import annotations

from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sacroml.attacks.attribute_attack import AttributeAttack
from sacroml.attacks.likelihood_attack import LIRAAttack
from sacroml.attacks.structural_attack import StructuralAttack
from sacroml.attacks.target import Target
from sacroml.attacks.worst_case_attack import WorstCaseAttack

output_dir: str = "output_pca"
target_dir: str = "target_pca"

RANDOM_STATE: int | None = 1
N_COMPONENTS: int = 5
N_SAMPLES: int = 200
N_FEATURES: int = 100
TEST_SIZE: float = 0.2


def test_sklearn() -> None:
    """Test scikit-learn PCA handling."""
    X, _ = make_regression(
        n_samples=N_SAMPLES, n_features=N_FEATURES, random_state=RANDOM_STATE
    )

    X_train, X_test = train_test_split(
        X, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    model.fit(X_train)

    target = Target(
        model=model,
        model_task="preprocessor",
        dataset_name="synthetic",
        X_train=X_train,
        X_test=X_test,
    )

    for i in range(N_FEATURES):
        target.add_feature(name=f"f_{i}", indices=[i], encoding="float")

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
    assert not output

    # Test structural attack
    attack = StructuralAttack(output_dir=output_dir)
    output = attack.attack(target)
    assert not output

    # Test attribute attack
    attack = AttributeAttack(n_cpu=2, output_dir=output_dir)
    output = attack.attack(target)
    assert not output

    # Test LiRA attack
    attack = LIRAAttack(n_shadow_models=64, output_dir=output_dir)
    output = attack.attack(target)
    assert not output
