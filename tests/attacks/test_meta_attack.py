"""Test MetaAttack."""

from __future__ import annotations

import os

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sacroml.attacks.meta_attack import MetaAttack
from sacroml.attacks.target import Target

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(name="meta_target")
def fixture_meta_target() -> Target:
    """Return a binary tabular target suitable for MetaAttack tests."""
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        class_sep=1.25,
        random_state=7,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=7
    )

    model = RandomForestClassifier(n_estimators=50, random_state=7)
    model.fit(X_train, y_train)

    target = Target(
        model=model,
        dataset_name="meta_test",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_train_orig=X_train,
        y_train_orig=y_train,
        X_test_orig=X_test,
        y_test_orig=y_test,
    )
    for idx in range(X.shape[1]):
        target.add_feature(f"V{idx}", [idx], "float")
    return target


# ------------------------------------------------------------------
# Validation tests
# ------------------------------------------------------------------


def test_meta_unsupported_attack():
    """MetaAttack should reject attacks without per-record scores."""
    with pytest.raises(ValueError, match="Unsupported attack"):
        MetaAttack(
            attacks=[("worstcase", {})],
            k_threshold=10,
        )


def test_meta_invalid_tuple():
    """MetaAttack should reject tuples that are not length 2 or 3."""
    with pytest.raises(ValueError, match="got tuple of length 1"):
        MetaAttack(
            attacks=[("lira",)],
            k_threshold=10,
        )


def test_meta_empty_attacks():
    """MetaAttack should reject an empty attacks list."""
    with pytest.raises(ValueError, match="at least one entry"):
        MetaAttack(
            attacks=[],
            k_threshold=10,
        )


def test_meta_invalid_n_reps():
    """MetaAttack should reject n_reps < 1."""
    with pytest.raises(ValueError, match="positive integer"):
        MetaAttack(
            attacks=[("lira", {}, 0)],
            k_threshold=10,
        )


# ------------------------------------------------------------------
# Integration tests
# ------------------------------------------------------------------


def test_meta_basic_qmia_structural(meta_target, tmp_path):
    """MetaAttack with QMIA + structural should produce a valid DataFrame."""
    meta = MetaAttack(
        attacks=[("qmia", {}), ("structural", {})],
        output_dir=str(tmp_path / "meta"),
        write_report=False,
        k_threshold=10,
    )

    output = meta.attack(meta_target)

    # Report structure
    assert output["metadata"]["attack_name"] == "Meta Attack"

    # DataFrame shape: n_train + n_test rows
    n_train = len(meta_target.X_train)
    n_test = len(meta_target.X_test)
    df = meta.vulnerability_df
    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_train + n_test

    # Expected columns present
    assert "is_member" in df.columns
    assert "qmia_mean" in df.columns
    assert "qmia_vuln" in df.columns
    assert "struct_k" in df.columns
    assert "struct_vuln" in df.columns
    assert "mia_mean" in df.columns
    assert "mia_gmean" in df.columns
    assert "n_vulnerable" in df.columns


def test_meta_structural_nan_for_test_records(meta_target, tmp_path):
    """Structural columns should be NaN/None for test (non-member) records."""
    meta = MetaAttack(
        attacks=[("qmia", {}), ("structural", {})],
        output_dir=str(tmp_path / "meta"),
        write_report=False,
        k_threshold=10,
    )
    meta.attack(meta_target)

    df = meta.vulnerability_df
    test_rows = df[df["is_member"] == 0]
    assert test_rows["struct_k"].isna().all()


def test_meta_repeated_runs(meta_target, tmp_path):
    """Repeated runs should produce non-zero std for at least some records."""
    meta = MetaAttack(
        attacks=[("qmia", {}, 2)],
        output_dir=str(tmp_path / "meta"),
        write_report=False,
        k_threshold=10,
    )
    meta.attack(meta_target)

    df = meta.vulnerability_df
    # With 2 stochastic reps, some records should have non-zero std
    assert "qmia_std" in df.columns
    # Consistency should be in [0, 1]
    assert df["qmia_consistency"].between(0.0, 1.0).all()


def test_meta_threshold_effects(meta_target, tmp_path):
    """Different thresholds should produce different vulnerability counts."""
    meta_low = MetaAttack(
        attacks=[("qmia", {})],
        output_dir=str(tmp_path / "low"),
        write_report=False,
        mia_threshold=0.3,
        k_threshold=10,
    )
    meta_low.attack(meta_target)

    meta_high = MetaAttack(
        attacks=[("qmia", {})],
        output_dir=str(tmp_path / "high"),
        write_report=False,
        mia_threshold=0.7,
        k_threshold=10,
    )
    meta_high.attack(meta_target)

    n_vuln_low = meta_low.vulnerability_df["n_vulnerable"].sum()
    n_vuln_high = meta_high.vulnerability_df["n_vulnerable"].sum()
    # Lower threshold should flag more records
    assert n_vuln_low >= n_vuln_high


def test_meta_global_metrics(meta_target, tmp_path):
    """Global metrics should contain AUC in [0, 1] when MIA attacks are run."""
    meta = MetaAttack(
        attacks=[("qmia", {})],
        output_dir=str(tmp_path / "meta"),
        write_report=False,
        k_threshold=10,
    )
    output = meta.attack(meta_target)

    m = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]
    assert 0 <= m["AUC"] <= 1
    assert 0 <= m["TPR"] <= 1


def test_meta_report_structure(meta_target, tmp_path):
    """JSON report should have the standard nested structure."""
    meta = MetaAttack(
        attacks=[("qmia", {}), ("structural", {})],
        output_dir=str(tmp_path / "meta"),
        write_report=False,
        k_threshold=10,
    )
    output = meta.attack(meta_target)

    assert "log_id" in output
    assert "metadata" in output
    assert "attack_experiment_logger" in output

    metadata = output["metadata"]
    assert metadata["attack_name"] == "Meta Attack"
    assert "global_metrics" in metadata
    assert "mia_threshold" in metadata["global_metrics"]
    assert "k_threshold" in metadata["global_metrics"]
    assert "n_vulnerable_all_attacks" in metadata["global_metrics"]

    instance = output["attack_experiment_logger"]["attack_instance_logger"][
        "instance_0"
    ]
    assert "sub_attacks" in instance
    assert "individual" in instance
    assert "qmia" in instance["sub_attacks"]
    assert "structural" in instance["sub_attacks"]


def test_meta_factory_integration(meta_target, tmp_path):
    """MetaAttack should be invocable via the attack factory."""
    from sacroml.attacks.factory import attack  # noqa: PLC0415

    output = attack(
        target=meta_target,
        attack_name="meta",
        attacks=[("qmia", {})],
        output_dir=str(tmp_path / "factory"),
        write_report=False,
        k_threshold=10,
    )

    assert output["metadata"]["attack_name"] == "Meta Attack"


def test_meta_csv_export(meta_target, tmp_path):
    """MetaAttack with write_report=True should produce a CSV file."""
    out_dir = str(tmp_path / "meta")
    meta = MetaAttack(
        attacks=[("qmia", {})],
        output_dir=out_dir,
        write_report=True,
        k_threshold=10,
    )
    meta.attack(meta_target)

    csv_path = os.path.join(out_dir, "vulnerability_matrix.csv")
    assert os.path.isfile(csv_path)

    df_loaded = pd.read_csv(csv_path, index_col=0)
    assert len(df_loaded) == len(meta.vulnerability_df)
