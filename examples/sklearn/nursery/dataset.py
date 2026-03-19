"""Example dataset handler for the nursery dataset.

Scikit-learn datasets must implement `sacroml.attacks.data.SklearnDataHandler`.
"""

from collections.abc import Sequence

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sacroml.attacks.data import SklearnDataHandler

random_state = 1


class Nursery(SklearnDataHandler):
    """Nursery dataset handler."""

    def __init__(self) -> None:
        """Create and process a local nursery-like dataset."""
        self.X_orig, self.y_orig, self.feature_names = _make_local_nursery_data()

        # Process dataset
        self.label_enc = LabelEncoder()
        self.feature_enc = OneHotEncoder()
        self.X = self.feature_enc.fit_transform(self.X_orig).toarray()
        self.y = self.label_enc.fit_transform(self.y_orig)

        # Feature encoding information (only required for attribute inference)
        self.feature_indices = [
            [0, 1, 2],  # parents
            [3, 4, 5, 6, 7],  # has_nurs
            [8, 9, 10, 11],  # form
            [12, 13, 14, 15],  # children
            [16, 17, 18],  # housing
            [19, 20],  # finance
            [21, 22, 23],  # social
            [24, 25, 26],  # health
        ]

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.X)

    def get_raw_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the original raw data arrays."""
        return self.X_orig, self.y_orig

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the processed data arrays."""
        return self.X, self.y

    def get_subset(
        self, X: np.ndarray, y: np.ndarray, indices: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a subset of the data."""
        return X[indices], y[indices]

    def get_train_test_indices(self) -> tuple[Sequence[int], Sequence[int]]:
        """Return train and test set indices."""
        indices = range(len(self))
        train, test = train_test_split(
            indices, test_size=0.5, stratify=self.y, random_state=random_state
        )
        return train, test


def _make_local_nursery_data(
    n_samples: int = 6000, random_state: int = 1
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create deterministic nursery-like categorical data locally."""
    feature_names: list[str] = [
        "parents",
        "has_nurs",
        "form",
        "children",
        "housing",
        "finance",
        "social",
        "health",
    ]
    categories: list[list[str]] = [
        ["usual", "pretentious", "great_pret"],
        ["proper", "less_proper", "improper", "critical", "very_crit"],
        ["complete", "completed", "incomplete", "foster"],
        ["1", "2", "3", "more"],
        ["convenient", "less_conv", "critical"],
        ["convenient", "inconv"],
        ["nonprob", "slightly_prob", "problematic"],
        ["recommended", "priority", "not_recom"],
    ]
    class_names = np.asarray(
        ["not_recom", "recommend", "very_recom", "priority", "spec_prior"],
        dtype=str,
    )

    x_num, y_num = make_classification(
        n_samples=n_samples,
        n_features=len(feature_names),
        n_informative=6,
        n_redundant=0,
        n_repeated=0,
        n_classes=len(class_names),
        n_clusters_per_class=1,
        class_sep=1.2,
        random_state=random_state,
    )
    x_cat = np.empty((n_samples, len(feature_names)), dtype=object)
    for idx, values in enumerate(categories):
        col = x_num[:, idx]
        thresholds = np.quantile(col, np.linspace(0, 1, len(values) + 1)[1:-1])
        bins = np.digitize(col, thresholds)
        x_cat[:, idx] = np.asarray(values, dtype=str)[bins]

    y = class_names[y_num]
    return x_cat.astype(str), y.astype(str), feature_names
