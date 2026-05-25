"""Cross-attack canary detection tests.

Each MIA attack should flag deliberately memorised "canary" training rows
as members at higher confidence than genuine non-members. The fixture
``canary_target`` (defined in ``tests/conftest.py``) builds a target whose
training set contains label-flipped rows that sit on the decision
boundary; a non-bagging RandomForestClassifier memorises them, blowing
up their per-record MIA signal.

WorstCaseAttack is intentionally excluded for now. Its per-record output
indexes into an internal ``train_test_split`` of the combined train+test
predictions, not into the original training set, so canary indices do
not map directly. A WorstCase canary test is a follow-up.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from sacroml.attacks.likelihood_attack import LIRAAttack
from sacroml.attacks.qmia_attack import QMIAAttack

# Per-attack thresholds. Shadow-model attacks (LiRA) need looser bounds
# than QMIA's quantile regression because per-canary variance is higher.
CANARY_PARAMS = [
    pytest.param(
        QMIAAttack,
        {"random_state": 0},
        "member_prob",
        0.90,
        7,
        id="qmia",
    ),
    pytest.param(
        LIRAAttack,
        {"n_shadow_models": 20},
        "score",
        0.85,
        6,
        id="lira",
    ),
]


@pytest.mark.parametrize(
    ("attack_cls", "attack_kwargs", "score_field", "auc_threshold", "canary_threshold"),
    CANARY_PARAMS,
)
def test_attack_predicts_canaries(
    attack_cls,
    attack_kwargs,
    score_field,
    auc_threshold,
    canary_threshold,
    canary_target,
    tmp_path,
):
    """Attack flags memorised canaries above genuine non-members."""
    target, canary_idx, n_train = canary_target
    n_canaries = len(canary_idx)

    attack_obj = attack_cls(
        output_dir=str(tmp_path / f"canary_{attack_cls.__name__}"),
        write_report=False,
        report_individual=True,
        **attack_kwargs,
    )
    output = attack_obj.attack(target)

    individual = output["attack_experiment_logger"]["attack_instance_logger"][
        "instance_0"
    ]["individual"]
    member_prob = np.asarray(individual[score_field])

    canary_mp = member_prob[canary_idx]
    test_mp = member_prob[n_train:]

    # AUC of canaries (positives) vs genuine non-members (negatives).
    # > auc_threshold confirms the attack flags memorised rows correctly.
    y_score = np.concatenate([canary_mp, test_mp])
    y_true = np.concatenate([np.ones_like(canary_mp), np.zeros_like(test_mp)])
    canary_vs_test_auc = roc_auc_score(y_true, y_score)
    assert canary_vs_test_auc > auc_threshold, (
        f"{attack_cls.__name__} failed canary AUC: "
        f"AUC={canary_vs_test_auc:.3f} (threshold {auc_threshold})"
    )

    # Most canaries should land above the 90th percentile of test scores.
    test_p90 = np.percentile(test_mp, 90)
    n_above_p90 = int((canary_mp > test_p90).sum())
    assert n_above_p90 >= canary_threshold, (
        f"{attack_cls.__name__}: only {n_above_p90}/{n_canaries} canaries "
        f"exceed test 90th percentile ({test_p90:.3f}); "
        f"canary scores: {sorted(canary_mp.tolist())}"
    )
