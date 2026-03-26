"""Nursery dataset handler for scikit-learn models."""

from __future__ import annotations

import types
from collections.abc import Sequence

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sacroml.attacks.data import SklearnDataHandler

random_state = 1


def _generate_nursery_data(n_samples=2000, random_state=1):
    """Generate synthetic categorical data mimicking the nursery dataset.

    Uses make_classification to create a learnable classification problem,
    then discretises continuous features into categories matching the
    OpenML nursery dataset (data_id=26) structure so that one-hot encoding
    yields the same column layout.
    """
    feature_specs = [
        ("parents", ["great_pret", "pretentious", "usual"]),
        (
            "has_nurs",
            ["critical", "less_proper", "proper", "slightly_prob", "very_crit"],
        ),
        ("form", ["complete", "foster", "other", "others"]),
        ("children", ["1", "2", "3", "more"]),
        ("housing", ["convenient", "less_proper", "slightly_prob"]),
        ("finance", ["convenient", "inconv"]),
        ("social", ["non_prob", "slightly_prob", "very_recom"]),
        ("health", ["not_recom", "priority", "recommended"]),
    ]
    target_classes = ["not_recom", "priority", "spec_prior", "very_recom"]

    n_features = len(feature_specs)
    n_classes = len(target_classes)

    X_cont, y_int = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=random_state,
    )

    # Discretise each continuous feature into categories via percentile binning
    feature_names = []
    columns = []
    for i, (name, categories) in enumerate(feature_specs):
        feature_names.append(name)
        n_cats = len(categories)
        percentiles = np.linspace(0, 100, n_cats + 1)[1:-1]
        bins = np.percentile(X_cont[:, i], percentiles)
        bin_indices = np.digitize(X_cont[:, i], bins)
        columns.append(np.array([categories[idx] for idx in bin_indices]))

    data = np.column_stack(columns)
    target = np.array([target_classes[idx] for idx in y_int])

    return types.SimpleNamespace(data=data, target=target, feature_names=feature_names)


class Nursery(SklearnDataHandler):
    """Nursery dataset handler."""

    def __init__(self) -> None:
        """Fetch and process the nursery dataset."""
        # Get original dataset
        nursery_data = _generate_nursery_data()
        self.X_orig = np.asarray(nursery_data.data, dtype=str)
        self.y_orig = np.asarray(nursery_data.target, dtype=str)

        # Use only a sample to speed testing
        self.X_orig, _, self.y_orig, _ = train_test_split(
            self.X_orig,
            self.y_orig,
            test_size=0.95,
            stratify=self.y_orig,
            random_state=random_state,
        )

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
        self.feature_names = nursery_data.feature_names

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.X)

    def get_raw_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the original raw data arrays."""
        return self.X_orig, self.y_orig  # pragma: no cover

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
