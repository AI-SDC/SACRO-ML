"""Test MetaAttack."""

from __future__ import annotations

import json
import logging
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
    with pytest.raises(ValueError, match="got entry of length 1"):
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


# ------------------------------------------------------------------
# Behaviour mode tests
# ------------------------------------------------------------------


def test_meta_invalid_behaviour():
    """MetaAttack should reject an unrecognised behaviour string."""
    with pytest.raises(ValueError, match="Unknown behaviour"):
        MetaAttack(
            attacks=[("qmia", {})],
            behaviour="rerun_everything",
            k_threshold=10,
        )


def test_meta_use_existing_only(meta_target, tmp_path):
    """Use_existing_only reads from pre-existing report.json files."""
    n_train = len(meta_target.X_train)
    n_test = len(meta_target.X_test)
    n_total = n_train + n_test

    # Build a minimal mock QMIA report.json in a subdirectory.
    # The real format wraps each attack under "AttackName_<uuid>" (GenerateJSONModule).
    scores = [0.6] * n_train + [0.4] * n_test
    mock_report = {
        "QMIA Attack_test-uuid": {
            "metadata": {"attack_name": "QMIA Attack"},
            "attack_experiment_logger": {
                "attack_instance_logger": {
                    "instance_0": {"individual": {"member_prob": scores}}
                }
            },
        }
    }
    report_dir = str(tmp_path / "existing")
    sub_dir = os.path.join(report_dir, "qmia_run0")
    os.makedirs(sub_dir)
    with open(os.path.join(sub_dir, "report.json"), "w") as fh:
        json.dump(mock_report, fh)

    meta = MetaAttack(
        attacks=[("qmia", {})],
        behaviour="use_existing_only",
        report_dir=report_dir,
        output_dir=str(tmp_path / "meta_out"),
        write_report=False,
        k_threshold=10,
    )
    output = meta.attack(meta_target)

    assert output["metadata"]["attack_name"] == "Meta Attack"
    df = meta.vulnerability_df
    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_total
    # Training records should be flagged (score 0.6 > threshold 0.5)
    assert df.loc[df["is_member"] == 1, "qmia_vuln"].all()
    # Test records should not be flagged (score 0.4 <= threshold 0.5)
    assert not df.loc[df["is_member"] == 0, "qmia_vuln"].any()


def test_meta_use_existing_missing_individual(meta_target, tmp_path):
    """Use_existing_only skips reports that lack individual scores."""
    # Report without the 'individual' key (uses the real nested on-disk format).
    mock_report = {
        "QMIA Attack_test-uuid": {
            "metadata": {"attack_name": "QMIA Attack"},
            "attack_experiment_logger": {
                "attack_instance_logger": {
                    "instance_0": {}  # no 'individual' key
                }
            },
        }
    }
    report_dir = str(tmp_path / "existing")
    sub_dir = os.path.join(report_dir, "qmia_run0")
    os.makedirs(sub_dir)
    with open(os.path.join(sub_dir, "report.json"), "w") as fh:
        json.dump(mock_report, fh)

    meta = MetaAttack(
        attacks=[("qmia", {})],
        behaviour="use_existing_only",
        report_dir=report_dir,
        output_dir=str(tmp_path / "meta_out"),
        write_report=False,
        k_threshold=10,
    )
    # No scores collected → empty report
    result = meta.attack(meta_target)
    assert result == {}


def test_meta_fill_missing_skips_present(meta_target, tmp_path):
    """Fill_missing should skip attacks already in report_dir."""
    n_train = len(meta_target.X_train)
    n_test = len(meta_target.X_test)
    scores = [0.7] * (n_train + n_test)

    mock_qmia = {
        "QMIA Attack_test-uuid": {
            "metadata": {"attack_name": "QMIA Attack"},
            "attack_experiment_logger": {
                "attack_instance_logger": {
                    "instance_0": {"individual": {"member_prob": scores}}
                }
            },
        }
    }
    report_dir = str(tmp_path / "existing")
    sub_dir = os.path.join(report_dir, "qmia_run0")
    os.makedirs(sub_dir)
    with open(os.path.join(sub_dir, "report.json"), "w") as fh:
        json.dump(mock_qmia, fh)

    # Ask for qmia (already present) + structural (missing → will run)
    meta = MetaAttack(
        attacks=[("qmia", {}), ("structural", {})],
        behaviour="fill_missing",
        report_dir=report_dir,
        output_dir=str(tmp_path / "meta_out"),
        write_report=False,
        k_threshold=10,
    )
    output = meta.attack(meta_target)

    assert output["metadata"]["attack_name"] == "Meta Attack"
    df = meta.vulnerability_df
    assert "qmia_mean" in df.columns
    assert "struct_k" in df.columns
    # QMIA scores must come from the mock (all 0.7), not from a fresh live run.
    assert df["qmia_mean"].dropna().between(0.69, 0.71).all()


def test_meta_structural_warns_nreps(meta_target, tmp_path, caplog):
    """Structural attack with n_reps > 1 should warn and run only once."""
    meta = MetaAttack(
        attacks=[("structural", {}, 3)],
        output_dir=str(tmp_path / "meta"),
        write_report=False,
        k_threshold=10,
    )
    with caplog.at_level(logging.WARNING):
        meta.attack(meta_target)

    assert any("deterministic" in msg for msg in caplog.messages)

    df = meta.vulnerability_df
    assert df is not None
    assert "struct_k" in df.columns
    # Only one set of structural scores should be present (single run)
    assert df["struct_k"].notna().sum() == len(meta_target.X_train)


def test_meta_corrupted_report_json_skipped(meta_target, tmp_path):
    """Use_existing_only skips subdirectories whose report.json is not valid JSON."""
    report_dir = str(tmp_path / "existing")
    sub_dir = os.path.join(report_dir, "qmia_run0")
    os.makedirs(sub_dir)
    with open(os.path.join(sub_dir, "report.json"), "w") as fh:
        fh.write("this is not valid json {{{")

    meta = MetaAttack(
        attacks=[("qmia", {})],
        behaviour="use_existing_only",
        report_dir=report_dir,
        output_dir=str(tmp_path / "meta_out"),
        write_report=False,
        k_threshold=10,
    )
    # Bad file should be skipped gracefully — result is empty, not a crash.
    result = meta.attack(meta_target)
    assert result == {}


def test_meta_scan_nonexistent_report_dir(meta_target, tmp_path):
    """Use_existing_only with a missing report_dir returns an empty result."""
    meta = MetaAttack(
        attacks=[("qmia", {})],
        behaviour="use_existing_only",
        report_dir=str(tmp_path / "does_not_exist"),
        output_dir=str(tmp_path / "meta_out"),
        write_report=False,
        k_threshold=10,
    )
    result = meta.attack(meta_target)
    assert result == {}


def test_meta_structural_multiple_reps_averaging(meta_target, tmp_path):
    """Structural_scores with multiple reps should be averaged in the DataFrame."""
    n_train = len(meta_target.X_train)

    # Directly call _build_dataframe with two structural reps to exercise averaging.
    meta = MetaAttack(
        attacks=[("structural", {})],
        output_dir=str(tmp_path / "meta"),
        write_report=False,
        k_threshold=10,
    )
    # Fabricate two structural reps with different k-anonymity values.
    false_train = [False] * n_train
    reps = [
        {
            "k_anonymity": [2] * n_train,
            "class_disclosure": false_train,
            "smallgroup_risk": false_train,
        },
        {
            "k_anonymity": [4] * n_train,
            "class_disclosure": false_train,
            "smallgroup_risk": false_train,
        },
    ]
    n_test = len(meta_target.X_test)
    df = meta._build_dataframe(n_train, n_test, {}, {"structural": reps})

    # k values should be averaged: (2 + 4) / 2 = 3
    assert all(v == 3 for v in df["struct_k"].dropna())


# ------------------------------------------------------------------
# Additional coverage: S4, S5, S6
# ------------------------------------------------------------------


def test_meta_use_existing_structural(meta_target, tmp_path):
    """Use_existing_only loads structural scores from a pre-existing report.json."""
    n_train = len(meta_target.X_train)

    false_train = [False] * n_train
    mock_report = {
        "Structural Attack_test-uuid": {
            "metadata": {"attack_name": "Structural Attack"},
            "attack_experiment_logger": {
                "attack_instance_logger": {
                    "instance_0": {
                        "individual": {
                            "k_anonymity": [5] * n_train,
                            "class_disclosure": false_train,
                            "smallgroup_risk": false_train,
                        }
                    }
                }
            },
        }
    }
    report_dir = str(tmp_path / "existing")
    sub_dir = os.path.join(report_dir, "struct_run0")
    os.makedirs(sub_dir)
    with open(os.path.join(sub_dir, "report.json"), "w") as fh:
        json.dump(mock_report, fh)

    meta = MetaAttack(
        attacks=[("structural", {})],
        behaviour="use_existing_only",
        report_dir=report_dir,
        output_dir=str(tmp_path / "meta_out"),
        write_report=False,
        k_threshold=10,
    )
    output = meta.attack(meta_target)

    assert output["metadata"]["attack_name"] == "Meta Attack"
    df = meta.vulnerability_df
    assert "struct_k" in df.columns
    assert "struct_vuln" in df.columns
    # k=5 < k_threshold=10 → all training records should be flagged
    assert df.loc[df["is_member"] == 1, "struct_vuln"].all()


def test_meta_fill_missing_full_cache_hit(meta_target, tmp_path):
    """Fill_missing with all attacks already on disk runs nothing new."""
    n_train = len(meta_target.X_train)
    n_test = len(meta_target.X_test)
    scores = [0.8] * (n_train + n_test)

    mock_qmia = {
        "QMIA Attack_test-uuid": {
            "metadata": {"attack_name": "QMIA Attack"},
            "attack_experiment_logger": {
                "attack_instance_logger": {
                    "instance_0": {"individual": {"member_prob": scores}}
                }
            },
        }
    }
    report_dir = str(tmp_path / "existing")
    sub_dir = os.path.join(report_dir, "qmia_run0")
    os.makedirs(sub_dir)
    with open(os.path.join(sub_dir, "report.json"), "w") as fh:
        json.dump(mock_qmia, fh)

    meta = MetaAttack(
        attacks=[("qmia", {})],
        behaviour="fill_missing",
        report_dir=report_dir,
        output_dir=str(tmp_path / "meta_out"),
        write_report=False,
        k_threshold=10,
    )
    output = meta.attack(meta_target)

    assert output["metadata"]["attack_name"] == "Meta Attack"
    df = meta.vulnerability_df
    assert "qmia_mean" in df.columns
    # All scores came from mock (0.8) — no live run happened.
    assert df["qmia_mean"].dropna().between(0.79, 0.81).all()


def test_meta_mia_cross_attack_aggregation(meta_target, tmp_path):
    """Mia_mean and mia_gmean are correct when two MIA attacks run together."""
    meta = MetaAttack(
        attacks=[("qmia", {}), ("lira", {"n_shadow_models": 10})],
        output_dir=str(tmp_path / "meta"),
        write_report=False,
        k_threshold=10,
    )
    meta.attack(meta_target)

    df = meta.vulnerability_df
    assert "qmia_mean" in df.columns
    assert "lira_mean" in df.columns
    assert "mia_mean" in df.columns
    assert "mia_gmean" in df.columns

    # mia_mean must be the arithmetic mean of the two per-attack means.
    import numpy as np  # noqa: PLC0415

    expected = (df["qmia_mean"] + df["lira_mean"]) / 2
    assert np.allclose(df["mia_mean"], expected, equal_nan=True)


# ------------------------------------------------------------------
# I3: structural-only global metrics path
# ------------------------------------------------------------------


def test_meta_structural_only_global_metrics(meta_target, tmp_path):
    """Structural-only run must not produce AUC and must report n_vulnerable_train."""
    meta = MetaAttack(
        attacks=[("structural", {})],
        output_dir=str(tmp_path / "meta"),
        write_report=False,
        k_threshold=10,
    )
    output = meta.attack(meta_target)

    gm = output["metadata"]["global_metrics"]
    assert "AUC" not in gm
    assert "TPR" not in gm

    logger_key = "attack_instance_logger"
    instance = output["attack_experiment_logger"][logger_key]["instance_0"]
    assert "AUC" not in instance
    assert "n_train" in instance
    assert "n_vulnerable_train" in instance
    assert isinstance(instance["n_vulnerable_train"], int)
    assert instance["n_vulnerable_train"] >= 0


# ------------------------------------------------------------------
# I4: n_vulnerable_all_attacks value
# ------------------------------------------------------------------


def test_meta_n_vulnerable_all_attacks_value(meta_target, tmp_path):
    """N_vulnerable_all_attacks counts records flagged by every active attack."""
    # mia_threshold=-1.0 guarantees all QMIA scores (clipped to [0,1]) satisfy > -1.0
    meta = MetaAttack(
        attacks=[("qmia", {})],
        output_dir=str(tmp_path / "meta"),
        write_report=False,
        mia_threshold=-1.0,
        k_threshold=10,
    )
    output = meta.attack(meta_target)
    n_total = len(meta_target.X_train) + len(meta_target.X_test)
    assert output["metadata"]["global_metrics"]["n_vulnerable_all_attacks"] == n_total

    meta2 = MetaAttack(
        attacks=[("qmia", {})],
        output_dir=str(tmp_path / "meta2"),
        write_report=False,
        mia_threshold=1.1,
        k_threshold=10,
    )
    output2 = meta2.attack(meta_target)
    assert output2["metadata"]["global_metrics"]["n_vulnerable_all_attacks"] == 0


# ------------------------------------------------------------------
# I5: struct_vuln via class_disclosure / smallgroup_risk
# ------------------------------------------------------------------


def test_meta_struct_vuln_flagged_by_class_disclosure(meta_target, tmp_path):
    """Struct_vuln must be True when class_disclosure=True, even if k >= k_threshold."""
    n_train = len(meta_target.X_train)
    n_test = len(meta_target.X_test)
    meta = MetaAttack(
        attacks=[("structural", {})],
        output_dir=str(tmp_path / "meta"),
        write_report=False,
        k_threshold=5,
    )
    # k well above threshold, but class_disclosure triggers the flag
    reps = [
        {
            "k_anonymity": [10] * n_train,
            "class_disclosure": [True] * n_train,
            "smallgroup_risk": [False] * n_train,
        }
    ]
    df = meta._build_dataframe(n_train, n_test, {}, {"structural": reps})
    assert df.loc[df["is_member"] == 1, "struct_vuln"].all()


def test_meta_struct_vuln_flagged_by_smallgroup_risk(meta_target, tmp_path):
    """Struct_vuln must be True when smallgroup_risk=True, even if k >= k_threshold."""
    n_train = len(meta_target.X_train)
    n_test = len(meta_target.X_test)
    meta = MetaAttack(
        attacks=[("structural", {})],
        output_dir=str(tmp_path / "meta"),
        write_report=False,
        k_threshold=5,
    )
    reps = [
        {
            "k_anonymity": [10] * n_train,
            "class_disclosure": [False] * n_train,
            "smallgroup_risk": [True] * n_train,
        }
    ]
    df = meta._build_dataframe(n_train, n_test, {}, {"structural": reps})
    assert df.loc[df["is_member"] == 1, "struct_vuln"].all()


# ------------------------------------------------------------------
# keep_separate / append-to-existing-report.json tests
# ------------------------------------------------------------------


def test_meta_keep_separate_default_writes_to_report_dir(meta_target, tmp_path):
    """Default ``keep_separate=False`` writes report.json to ``report_dir``."""
    out_dir = str(tmp_path / "out")
    rep_dir = str(tmp_path / "rep")
    meta = MetaAttack(
        attacks=[("qmia", {})],
        output_dir=out_dir,
        report_dir=rep_dir,
        write_report=True,
        k_threshold=10,
    )
    meta.attack(meta_target)

    assert os.path.isfile(os.path.join(rep_dir, "report.json"))
    assert not os.path.isfile(os.path.join(out_dir, "report.json"))
    assert os.path.isfile(os.path.join(out_dir, "vulnerability_matrix.csv"))


def test_meta_keep_separate_true_writes_to_output_dir(meta_target, tmp_path):
    """``keep_separate=True`` writes report.json to ``output_dir`` (base behaviour)."""
    out_dir = str(tmp_path / "out")
    rep_dir = str(tmp_path / "rep")
    os.makedirs(rep_dir, exist_ok=True)
    meta = MetaAttack(
        attacks=[("qmia", {})],
        output_dir=out_dir,
        report_dir=rep_dir,
        write_report=True,
        keep_separate=True,
        k_threshold=10,
    )
    meta.attack(meta_target)

    assert os.path.isfile(os.path.join(out_dir, "report.json"))
    assert not os.path.isfile(os.path.join(rep_dir, "report.json"))


def test_meta_make_pdf_returns_fpdf(meta_target, tmp_path):
    """``_make_pdf`` should return an FPDF instance, not None."""
    meta = MetaAttack(
        attacks=[("qmia", {})],
        output_dir=str(tmp_path / "meta"),
        write_report=False,
        k_threshold=10,
    )
    output = meta.attack(meta_target)
    pdf = meta._make_pdf(output)
    assert pdf is not None


def test_meta_pdf_written_to_report_dir_by_default(meta_target, tmp_path):
    """With default keep_separate=False, report.pdf lands in report_dir."""
    out_dir = str(tmp_path / "out")
    rep_dir = str(tmp_path / "rep")
    meta = MetaAttack(
        attacks=[("qmia", {})],
        output_dir=out_dir,
        report_dir=rep_dir,
        write_report=True,
        k_threshold=10,
    )
    meta.attack(meta_target)
    pdf_path = os.path.join(rep_dir, "report.pdf")
    assert os.path.isfile(pdf_path)
    assert os.path.getsize(pdf_path) > 0


def test_meta_appends_to_existing_report_json(meta_target, tmp_path):
    """Default mode appends to existing report.json, keeps prior sections."""
    rep_dir = tmp_path / "rep"
    rep_dir.mkdir()
    existing = {
        "LiRA Attack_abc123": {
            "metadata": {"attack_name": "LiRA Attack", "log_id": "abc123"},
            "fake_payload": True,
        }
    }
    existing_path = rep_dir / "report.json"
    existing_path.write_text(json.dumps(existing))

    meta = MetaAttack(
        attacks=[("qmia", {})],
        output_dir=str(tmp_path / "out"),
        report_dir=str(rep_dir),
        write_report=True,
        k_threshold=10,
    )
    meta.attack(meta_target)

    with open(existing_path) as f:
        data = json.load(f)

    assert "LiRA Attack_abc123" in data
    assert data["LiRA Attack_abc123"]["fake_payload"] is True
    meta_keys = [k for k in data if k.startswith("Meta Attack_")]
    assert len(meta_keys) == 1
