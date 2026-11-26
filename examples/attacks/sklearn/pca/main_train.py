"""Example training a scikit-learn PCA model.

Note: In this example the data (`X_train`, `X_test`, etc.) is passed directly as
numpy arrays to the `sacroml.attacks.target.Target` constructor. In this case,
the data will be copied to the target directory along with the model.
An alternative is to implement `sacroml.attacks.data.SklearnDataHandler` and data
will instead be loaded at runtime when performing an audit/attack.
"""

import logging

from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sacroml.attacks.target import Target

output_dir: str = "target_pca"

RANDOM_STATE: int | None = 1
N_COMPONENTS: int = 5
N_SAMPLES: int = 200
N_FEATURES: int = 100
TEST_SIZE: float = 0.2


if __name__ == "__main__":
    logging.info("Creating synthetic data")
    X, _ = make_regression(
        n_samples=N_SAMPLES, n_features=N_FEATURES, random_state=RANDOM_STATE
    )

    logging.info("Splitting data into training and test sets")
    X_train, X_test = train_test_split(
        X, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    logging.info("Defining the model")
    model = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)

    logging.info("Training the model")
    model.fit(X_train)

    logging.info("Wrapping the model and data in a Target object")
    target = Target(
        model=model,
        model_task="preprocessor",
        dataset_name="synthetic",
        X_train=X_train,
        X_test=X_test,
    )

    logging.info("Writing Target object to directory: '%s'", output_dir)
    target.save(output_dir)
