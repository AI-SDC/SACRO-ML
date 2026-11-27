"""Minimal example training and auditing a scikit-learn PCA model.

Attack JSON and PDF reports will be created in the `output_dir/`
"""

from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sacroml.attacks.likelihood_attack import LIRAAttack
from sacroml.attacks.target import Target
from sacroml.attacks.threshold_attack import ThresholdAttack

output_dir: str = "output_pca"

RANDOM_STATE: int | None = 1
N_COMPONENTS: int = 5
N_SAMPLES: int = 200
N_FEATURES: int = 100
TEST_SIZE: float = 0.2


if __name__ == "__main__":
    # Load dataset
    X, _ = make_regression(
        n_samples=N_SAMPLES, n_features=N_FEATURES, random_state=RANDOM_STATE
    )
    X_train, X_test = train_test_split(
        X, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Fit model
    model = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    model.fit(X_train)

    # Wrap model and data
    target = Target(
        model=model,
        model_task="preprocessor",
        dataset_name="synthetic",
        X_train=X_train,
        X_test=X_test,
    )

    # Run a threshold reconstruction error attack
    attack = ThresholdAttack(output_dir=output_dir)
    attack.attack(target)

    # Run a LiRA attack using reconstruction errors
    attack = LIRAAttack(
        n_shadow_models=100, mode="online-carlini", output_dir=output_dir
    )
    attack.attack(target)
