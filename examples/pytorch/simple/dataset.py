"""An example synthetic dataset."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_data(
    n_features: int, n_classes: int, random_state: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a synthetic dataset for testing."""
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

    input_encoder = StandardScaler()
    X = input_encoder.fit_transform(X_orig)
    y = y_orig  # leave as labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, shuffle=True, random_state=random_state
    )

    X = np.asarray(X)
    y = np.asarray(y)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    return X, y, X_train, y_train, X_test, y_test
