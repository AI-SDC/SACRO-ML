"""Regression targets through the existing attack entry points."""

import json

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from sacroml.attacks import attribute_attack, utils
from sacroml.attacks.attribute_attack import AttributeAttack
from sacroml.attacks.instance_based_attack import InstanceBasedAttack
from sacroml.attacks.likelihood_attack import LIRAAttack
from sacroml.attacks.meta_attack import MetaAttack
from sacroml.attacks.model_sklearn import SklearnModel
from sacroml.attacks.qmia_attack import QMIAAttack
from sacroml.attacks.structural_attack import StructuralAttack, get_model_param_count
from sacroml.attacks.target import Target
from sacroml.attacks.worst_case_attack import WorstCaseAttack


@pytest.fixture
def regression_target():
    """Fit a regressor with continuous targets and unequal split sizes."""
    rng = np.random.default_rng(414)
    x = rng.normal(size=(400, 3))
    y = 2 * x[:, 0] + x[:, 1] ** 2 + rng.normal(size=400) * (0.2 + abs(x[:, 0]))
    model = DecisionTreeRegressor(random_state=3).fit(x[:140], y[:140])
    return Target(
        model=model, X_train=x[:140], y_train=y[:140], X_test=x[140:], y_test=y[140:]
    )


def instance(output):
    """Read the first reported attack instance."""
    return output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]


@pytest.mark.parametrize("column_labels", [False, True])
def test_regression_losses_and_generalisation_gap(column_labels):
    """MSE uses matching rows and test-minus-train error, not R-squared."""
    x = np.array([[0.0], [1.0]])
    model = SklearnModel(DummyRegressor(strategy="constant", constant=0).fit(x, [0, 0]))
    train_y = np.array([1.0, 2.0])
    test_y = np.array([3.0, 4.0])
    if column_labels:
        train_y, test_y = train_y[:, None], test_y[:, None]
    np.testing.assert_allclose(model.get_losses(x, train_y), [1, 4])
    assert model.get_generalisation_gap(x, train_y, x, test_y) == 10.0


def test_regression_labels_are_preserved(regression_target):
    """Continuous labels and non-member rows must not be remapped or dropped."""
    target = regression_target
    before_train = target.y_train.copy()
    before_test = target.y_test.copy()
    target.y_train = target.y_train[:, None]
    utils.check_and_update_dataset(target)
    np.testing.assert_array_equal(target.y_train, before_train)
    np.testing.assert_array_equal(target.y_test, before_test)
    assert len(target.X_test) == len(before_test)


def test_regression_rejects_multiple_outputs():
    """Multi-output predictions fail explicitly instead of mixing rows."""
    x = np.arange(12).reshape(6, 2)
    y = np.column_stack((x[:, 0], x[:, 1]))
    model = SklearnModel(LinearRegression().fit(x, y))
    with pytest.raises(ValueError, match="single output"):
        model.get_losses(x, y)
    target = Target(model=model, X_train=x, X_test=x, y_train=y, y_test=y)
    with pytest.raises(ValueError, match="single output"):
        utils.check_and_update_dataset(target)


def test_regression_target_roundtrip(regression_target, tmp_path):
    """Saved and reloaded targets retain numeric labels, detection and losses."""
    target = regression_target
    target.model = SklearnModel(LinearRegression().fit(target.X_train, target.y_train))
    expected = target.model.get_losses(target.X_test, target.y_test)
    target.save(str(tmp_path / "saved"))
    restored = Target()
    restored.load(str(tmp_path / "saved"))
    assert restored.model.is_regression
    np.testing.assert_allclose(
        restored.model.get_losses(restored.X_test, restored.y_test), expected
    )


@pytest.mark.parametrize("include_error", [False, True])
def test_worstcase_regression(regression_target, tmp_path, include_error, monkeypatch):
    """A real membership classifier consumes predictions and optional squared errors."""
    attack = WorstCaseAttack(
        output_dir=str(tmp_path),
        write_report=True,
        n_reps=2,
        n_dummy_reps=0,
        include_model_correct_feature=include_error,
        attack_model_params={"n_estimators": 10, "random_state": 1},
    )
    original = attack._prepare_attack_data
    captured = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result[0])
        return result

    monkeypatch.setattr(attack, "_prepare_attack_data", capture)
    output = attack.attack(regression_target)
    assert output["metadata"]["target_task"] == "regression"
    assert 0 <= instance(output)["AUC"] <= 1
    assert captured[0].shape == (400, 2 if include_error else 1)
    if include_error:
        assert instance(output)["AUC"] > 0.9
        assert instance(output)["yeom_tpr"] == 1.0
        assert instance(output)["yeom_fpr"] == 0.0
        np.testing.assert_allclose(
            captured[0][:140, 1],
            regression_target.model.get_losses(
                regression_target.X_train, regression_target.y_train
            ),
        )
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.pdf").exists()


@pytest.mark.parametrize("mode", ["offline", "offline-carlini", "online-carlini"])
def test_lira_regression(regression_target, tmp_path, mode):
    """All likelihood modes train regressor shadows and preserve numerical labels."""
    target = regression_target
    target.model = SklearnModel(LinearRegression().fit(target.X_train, target.y_train))
    original_labels = np.concatenate((target.y_train, target.y_test))
    attack = LIRAAttack(
        output_dir=str(tmp_path),
        write_report=False,
        n_shadow_models=12,
        mode=mode,
        report_individual=True,
    )
    output = attack.attack(target)
    values = instance(output)
    assert np.isfinite(values["AUC"])
    individual = values["individual"]
    np.testing.assert_array_equal(individual["label"], original_labels)
    expected = -np.log(target.model.get_losses(target.X_train, target.y_train) + 1e-16)
    np.testing.assert_allclose(individual["target_signal"][:140], expected)
    assert "target_logit" not in individual
    shadow, train, _ = utils.get_shadow_model(str(tmp_path / "shadow_models"), 0)
    assert shadow.is_regression
    assert len(train) == 140


def test_qmia_regression(regression_target, tmp_path):
    """Quantile thresholds use negative squared error and produce real record scores."""
    attack = QMIAAttack(
        output_dir=str(tmp_path),
        write_report=True,
        max_iter=40,
        alpha=0.1,
        report_individual=True,
    )
    output = attack.attack(regression_target)
    assert output.get("status") != "failed", output.get("fail_reason")
    values = instance(output)
    assert 0 <= values["AUC"] <= 1
    assert len(values["individual"]["member_prob"]) == 400
    individual = values["individual"]
    np.testing.assert_allclose(individual["score"][:140], 0.0)
    assert np.all(np.asarray(individual["score"])[140:] < 0)
    np.testing.assert_allclose(
        individual["margin"], np.asarray(individual["score"]) - individual["threshold"]
    )
    assert (tmp_path / "report.json").exists()


@pytest.mark.parametrize(
    "estimator",
    [
        DecisionTreeRegressor(random_state=0),
        RandomForestRegressor(n_estimators=3, random_state=0),
        LinearRegression(),
        KNeighborsRegressor(n_neighbors=2),
        make_pipeline(StandardScaler(), LinearRegression()),
    ],
)
def test_structural_regression(regression_target, tmp_path, estimator):
    """Report meaningful structural checks and explicitly identify unavailable ones."""
    target = regression_target
    target.model = SklearnModel(estimator.fit(target.X_train, target.y_train))
    attack = StructuralAttack(
        output_dir=str(tmp_path), write_report=True, report_individual=True
    )
    output = attack.attack(target)
    values = instance(output)
    assert values["class_disclosure_risk"] is None
    assert values["smallgroup_risk"] is None
    assert values["unnecessary_risk"] is None
    assert values["individual"]["class_disclosure"] == [None] * 140
    assert len(values["individual"]["k_anonymity"]) == 140
    assert "train_acc" not in values
    assert values["train_mse"] >= 0
    assert values["generalisation_gap"] == pytest.approx(
        values["test_mse"] - values["train_mse"]
    )
    assert (tmp_path / "report.pdf").exists()
    written = json.loads((tmp_path / "report.json").read_text())
    stored = instance(next(iter(written.values())))
    assert stored["class_disclosure_risk"] is None
    assert stored["test_mse"] == pytest.approx(values["test_mse"])
    if isinstance(estimator, KNeighborsRegressor):
        assert values["dof_risk"] is None


def test_regression_tree_parameter_count():
    """A regression leaf stores a value rather than a class distribution."""
    tree = DecisionTreeRegressor(max_depth=1).fit([[0], [1], [2], [3]], [0, 0, 2, 2])
    assert get_model_param_count(tree) == 4


def test_regression_mlp_parameter_count():
    """Count the regressor's actual weights and biases."""
    model = MLPRegressor(hidden_layer_sizes=(2,), solver="lbfgs", random_state=0).fit(
        [[0], [1], [2], [3]], [0, 1, 2, 3]
    )
    assert get_model_param_count(model) == 7


def test_attribute_regression_identifies_categorical_values(tmp_path):
    """Recover a contributing attribute but abstain when candidates tie."""
    x = np.array([[a, b] for a in (0.0, 1.0, 2.0) for b in (0.0, 1.0, 2.0)])
    y = 2 * x[:, 0] + 0.25
    target = Target(
        model=LinearRegression().fit(x, y),
        X_train=x,
        X_test=x,
        y_train=y,
        y_test=y,
        X_train_orig=x,
        X_test_orig=x,
        y_train_orig=y,
        y_test_orig=y,
    )
    target.add_feature("signal", [0], "int")
    target.add_feature("irrelevant", [1], "int")
    output = AttributeAttack(
        output_dir=str(tmp_path), write_report=True, n_cpu=1
    ).attack(target)
    results = instance(output)["categorical"]
    assert results[0]["train"][:2] == (9, 9)
    assert results[1]["train"][:2] == (0, 0)
    assert (tmp_path / "report.pdf").exists()


@pytest.mark.parametrize("value", [-2.0, 2.0])
def test_attribute_regression_numeric_bounds(value):
    """Nearest predictions bound relevant numeric features, including negative values."""  # noqa: E501
    x = np.array([[a, b] for a in np.linspace(-3, 3, 11) for b in (0.0, 1.0)])
    model = SklearnModel(LinearRegression().fit(x, 2 * x[:, 0]))
    sample = np.array([value, 0.5])
    assert attribute_attack._get_bounds_risk_for_sample(model, 0, -3, 3, sample)
    assert not attribute_attack._get_bounds_risk_for_sample(model, 1, 0, 1, sample)


def test_meta_regression_repeated_structural(regression_target, tmp_path):
    """Repeated structural results retain unassessed fields and combine known risk."""
    attack = MetaAttack(
        attacks=[("structural", {}, 2)], output_dir=str(tmp_path), write_report=False
    )
    output = attack.attack(regression_target)
    assert output["metadata"]["target_task"] == "regression"
    df = attack.vulnerability_df
    assert df["struct_cd"].isna().all()
    assert df["struct_sg"].isna().all()
    members = df[df["is_member"] == 1]
    assert (
        members["struct_vuln"].tolist()
        == (members["struct_k"] < attack.k_threshold).tolist()
    )


def test_instance_based_regression(regression_target, tmp_path):
    """Existing kNN regression support still exposes stored training rows."""
    target = regression_target
    target.model = SklearnModel(
        KNeighborsRegressor().fit(target.X_train, target.y_train)
    )
    output = InstanceBasedAttack(output_dir=str(tmp_path), write_report=False).attack(
        target
    )
    assert output
    assert output["metadata"]["target_task"] == "regression"


def test_qmia_nonfinite_regression_fails_explicitly(
    regression_target, tmp_path, monkeypatch
):
    """A model returning NaN cannot produce a successful risk report."""
    monkeypatch.setattr(
        regression_target.model, "get_losses", lambda x, _: np.full(len(x), np.nan)
    )
    output = QMIAAttack(output_dir=str(tmp_path), write_report=False).attack(
        regression_target
    )
    assert output["status"] == "failed"
    assert "non-finite" in output["fail_reason"]


def test_regression_pipeline_losses_and_clone(regression_target):
    """Pipelines retain preprocessing and task detection when cloned for shadows."""
    target = regression_target
    model = SklearnModel(
        make_pipeline(StandardScaler(), LinearRegression()).fit(
            target.X_train, target.y_train
        )
    )
    clone = model.clone()
    clone.fit(target.X_train, target.y_train)
    assert clone.is_regression
    np.testing.assert_allclose(
        clone.predict(target.X_test), model.predict(target.X_test)
    )
    expected = (model.predict(target.X_test) - target.y_test) ** 2
    np.testing.assert_allclose(clone.get_losses(target.X_test, target.y_test), expected)
