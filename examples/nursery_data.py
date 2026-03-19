"""Shared nursery-like synthetic data generator for examples."""

import numpy as np
from sklearn.datasets import make_classification


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
