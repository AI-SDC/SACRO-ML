"""Attribute inference attacks."""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
from fpdf import FPDF
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder

from sacroml.attacks import report
from sacroml.attacks.attack import Attack
from sacroml.attacks.target import Target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLOR_A: str = "#86bf91"  # training set plot colour
COLOR_B: str = "steelblue"  # testing set plot colour


class AttributeAttack(Attack):
    """Attribute inference attack."""

    def __init__(
        self,
        output_dir: str = "outputs",
        write_report: bool = True,
        n_cpu: int = max(1, mp.cpu_count() - 1),
    ) -> None:
        """Construct an object to execute an attribute inference attack.

        Parameters
        ----------
        n_cpu : int
            number of CPUs used to run the attack
        output_dir : str
            name of the directory where outputs are stored
        write_report : bool
            Whether to generate a JSON and PDF report.
        """
        super().__init__(output_dir=output_dir, write_report=write_report)
        self.n_cpu = n_cpu

    def __str__(self) -> str:
        """Return the name of the attack."""
        return "Attribute inference attack"

    def attack(self, target: Target) -> dict:
        """Run attribute inference attack.

        To be used when code has access to Target class and trained target model.

        Parameters
        ----------
        target : attacks.target.Target
            target is a Target class object

        Returns
        -------
        dict
            Attack report.
        """
        if target.model is None or not target.has_data():  # pragma: no cover
            logger.info("WARNING: AttributeAttack requires a loadable model.")
            return {}

        if target.n_features < 1 or not target.has_raw_data():  # pragma: no cover
            logger.info("WARNING: AttributeAttack requires features to be defined.")
            return {}

        logger.info("Running attribute inference attack")
        self.attack_metrics = _attribute_inference(target, self.n_cpu)
        output = self._make_report(target)
        self._write_report(output)
        return output

    def _get_attack_metrics_instances(self) -> dict:
        """Construct the instances metric calculated, during attacks."""
        attack_metrics_experiment = {}
        attack_metrics_instances = {}
        attack_metrics_instances["instance_0"] = self.attack_metrics
        attack_metrics_experiment["attack_instance_logger"] = attack_metrics_instances
        return attack_metrics_experiment

    def _make_pdf(self, output: dict) -> FPDF:
        """Create PDF report."""
        metadata: dict = output["metadata"]
        metrics: dict = output["attack_experiment_logger"]["attack_instance_logger"][
            "instance_0"
        ]
        path: str = metadata["attack_params"]["output_dir"]
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_xy(0, 0)
        report.title(pdf, "Attribute Inference Attack Report")
        report.subtitle(pdf, "Introduction")
        # Add attack parameters
        report.subtitle(pdf, "Metadata")
        for key, value in metadata["attack_params"].items():
            report.line(pdf, f"{key:>30s}: {str(value):30s}", font="courier")
        # Add attack results
        report.subtitle(pdf, "Metrics")
        # Categorical
        categ_rep: list[str] = report_categorical(metrics).split("\n")
        if len(categ_rep) > 1:
            report.line(pdf, "Categorical Features:", font="courier")
            for line in categ_rep:
                report.line(pdf, line, font="courier")
        # Quantatitive
        quant_rep: list[str] = report_quantitative(metrics).split("\n")
        if len(quant_rep) > 1:
            report.line(pdf, "Quantitative Features:", font="courier")
            for line in quant_rep:
                report.line(pdf, line, font="courier")
        # Add plots
        pdf.add_page()
        report.subtitle(pdf, "Plots")
        plot_categorical_risk(metrics, path)  # Create pngs
        plot_categorical_fraction(metrics, path)
        plot_quantitative_risk(metrics, path)
        graphs = ["cat_risk.png", "cat_frac.png", "quant_risk.png"]
        for graph in graphs:
            filename = os.path.join(path, graph)
            if os.path.exists(filename):
                pdf.image(filename, x=None, y=None, w=150, h=0, type="", link="")
                os.remove(filename)
        return pdf


def _unique_max(confidences: list[float], threshold: float) -> bool:
    """Return if there is a unique maximum confidence value above threshold."""
    if len(confidences) > 0:
        max_conf = np.max(confidences)
        if max_conf < threshold:
            return False
        unique, count = np.unique(confidences, return_counts=True)
        for u, c in zip(unique, count):
            if c == 1 and u == max_conf:
                return True
    return False


def _get_inference_data(  # pylint: disable=too-many-locals
    target: Target, feature_id: int, memberset: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a dataset of each sample with the attributes to test."""
    attack_feature: dict = target.features[feature_id]
    indices: list[int] = attack_feature["indices"]
    unique = np.unique(target.X_orig[:, feature_id])
    n_unique: int = len(unique)
    values = unique
    if attack_feature["encoding"] == "onehot":
        onehot_enc = OneHotEncoder()
        values = onehot_enc.fit_transform(unique.reshape(-1, 1)).toarray()
    # samples after encoding (e.g. one-hot)
    samples: np.ndarray = target.X_train
    # samples before encoding (e.g. str)
    samples_orig: np.ndarray = target.X_train_orig
    if not memberset:
        samples = target.X_test
        samples_orig = target.X_test_orig
    n_samples, x_dim = np.shape(samples)
    x_values = np.zeros((n_samples, n_unique, x_dim), dtype=np.float64)
    y_values = target.model.predict(samples)
    # for each sample to perform inference on
    # add each possible missing feature value
    for i, x in enumerate(samples):
        for j, value in enumerate(values):
            x_values[i][j] = np.copy(x)
            x_values[i][j][indices] = value
    _, counts = np.unique(samples_orig[:, feature_id], return_counts=True)
    baseline = (np.max(counts) / n_samples) * 100
    logger.debug("feature: %d x_values shape = %s", feature_id, np.shape(x_values))
    logger.debug("feature: %d y_values shape = %s", feature_id, np.shape(y_values))
    return x_values, y_values, baseline


def _infer(  # pylint: disable=too-many-locals
    target: Target,
    feature_id: int,
    threshold: float,
    memberset: bool,
) -> tuple[int, int, float, int, int]:
    """Infer attribute.

    For each possible missing value, compute the confidence scores and label
    with the target model; if the label matches the known target model label
    for the original sample, and the highest confidence score is unique, infer
    that attribute if the confidence score is greater than a threshold.
    """
    logger.debug("Attacking feature %d set %d", feature_id, int(memberset))
    correct: int = 0  # number of correct inferences made
    total: int = 0  # total number of inferences made
    x_values, y_values, baseline = _get_inference_data(target, feature_id, memberset)
    n_unique: int = len(x_values[1])
    samples = target.X_train if memberset else target.X_test
    for i, x in enumerate(x_values):  # each sample to perform inference on
        # get model confidence scores for all possible values for the sample
        confidence = target.model.predict_proba(x)
        conf = []  # confidences for each possible value with correct label
        attr = []  # features for each possible value with correct label
        # for each possible attribute value,
        # if the label matches the known target model label
        # then store the confidence score and the tested feature vector
        for j in range(n_unique):
            this_label = np.argmax(confidence[j])
            scores = confidence[j][this_label]
            if this_label == y_values[i]:
                conf.append(scores)
                attr.append(x[j])
        # is there is a unique maximum confidence score above threshold?
        if _unique_max(conf, threshold):
            total += 1
            if (attr[np.argmax(conf)] == samples[i]).all():
                correct += 1
    logger.debug("Finished attacking feature %d", feature_id)
    return correct, total, baseline, n_unique, len(samples)


def report_categorical(results: dict) -> str:
    """Return a string report of the categorical results."""
    results = results["categorical"]
    msg = ""
    for feature in results:
        name = feature["name"]
        _, _, _, n_unique, _ = feature["train"]
        msg += f"Attacking categorical feature {name} with {n_unique} unique values:\n"
        for tranche in ("train", "test"):
            correct, total, baseline, _, n_samples = feature[tranche]
            if total > 0:
                msg += (
                    f"Correctly inferred {(correct / total) * 100:.2f}% "
                    f"of {(total / n_samples) * 100:.2f}% of the {tranche} set; "
                    f"baseline: {baseline:.2f}%\n"
                )
            else:  # pragma: no cover
                msg += f"Unable to make any inferences of the {tranche} set\n"
    return msg


def report_quantitative(results: dict) -> str:
    """Return a string report of the quantitative results."""
    results = results["quantitative"]
    msg = ""
    for feature in results:
        msg += (
            f"{feature['name']}: "
            f"{feature['train']:.2f} train risk, "
            f"{feature['test']:.2f} test risk\n"
        )
    return msg


def plot_quantitative_risk(res: dict, path: str = "") -> None:
    """Generate a bar chart showing quantitative value risk scores.

    Parameters
    ----------
    res : dict
        Dictionary containing attribute inference attack results.
    path : str
        Directory to write plots.
    """
    results = res["quantitative"]
    if len(results) < 1:  # pragma: no cover
        return
    logger.debug("Plotting quantitative feature risk scores")
    x = np.arange(len(results))
    ya = []
    yb = []
    names = []
    for feature in results:
        names.append(feature["name"])
        ya.append(feature["train"] * 100)
        yb.append(feature["test"] * 100)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90)
    ax.set_ylim([0, 100])
    ax.bar(x + 0.2, ya, 0.4, align="center", color=COLOR_A, label="train set")
    ax.bar(x - 0.2, yb, 0.4, align="center", color=COLOR_B, label="test set")
    title = "Percentage of Set at Risk for Quantitative Attributes"
    ax.set_title(f"{res['name']}\n{title}")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(linestyle="dotted", linewidth=1)
    ax.legend(loc="best")
    plt.margins(y=0)
    plt.tight_layout()
    filename = os.path.join(path, "quant_risk.png")
    fig.savefig(filename, pad_inches=0, bbox_inches="tight")
    logger.debug("Saved quantitative risk plot: %s", filename)


def plot_categorical_risk(  # pylint: disable=too-many-locals
    res: dict, path: str = ""
) -> None:
    """Generate a bar chart showing categorical risk scores.

    Parameters
    ----------
    res : dict
        Dictionary containing attribute inference attack results.
    path : str
        Directory to write plots.
    """
    results: list[dict] = res["categorical"]
    if len(results) < 1:  # pragma: no cover
        return
    logger.debug("Plotting categorical feature risk scores")
    x: np.ndarray = np.arange(len(results))
    ya: list[float] = []
    yb: list[float] = []
    names: list[str] = []
    for feature in results:
        names.append(feature["name"])
        correct_a, total_a, baseline_a, _, _ = feature["train"]
        correct_b, total_b, baseline_b, _, _ = feature["test"]
        a = ((correct_a / total_a) * 100) - baseline_a if total_a > 0 else 0
        b = ((correct_b / total_b) * 100) - baseline_b if total_b > 0 else 0
        ya.append(a)
        yb.append(b)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90)
    ax.set_ylim([-100, 100])
    ax.bar(x + 0.2, ya, 0.4, align="center", color=COLOR_A, label="train set")
    ax.bar(x - 0.2, yb, 0.4, align="center", color=COLOR_B, label="test set")
    title: str = "Improvement Over Most Common Value Estimate"
    ax.set_title(f"{res['name']}\n{title}")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(linestyle="dotted", linewidth=1)
    ax.legend(loc="best")
    plt.margins(y=0)
    plt.tight_layout()
    filename = os.path.join(path, "cat_risk.png")
    fig.savefig(filename, pad_inches=0, bbox_inches="tight")
    logger.debug("Saved categorical risk plot: %s", filename)


def plot_categorical_fraction(  # pylint: disable=too-many-locals
    res: dict, path: str = ""
) -> None:
    """Generate a bar chart showing fraction of dataset inferred.

    Parameters
    ----------
    res : dict
        Dictionary containing attribute inference attack results.
    path : str
        Directory to write plots.
    """
    results: list[dict] = res["categorical"]
    if len(results) < 1:  # pragma: no cover
        return
    logger.debug("Plotting categorical feature tranche sizes")
    x: np.ndarray = np.arange(len(results))
    ya: list[float] = []
    yb: list[float] = []
    names: list[str] = []
    for feature in results:
        names.append(feature["name"])
        _, total_a, _, _, n_samples_a = feature["train"]
        _, total_b, _, _, n_samples_b = feature["test"]
        a = ((total_a / n_samples_a) * 100) if n_samples_a > 0 else 0
        b = ((total_b / n_samples_b) * 100) if n_samples_b > 0 else 0
        ya.append(a)
        yb.append(b)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90)
    ax.set_ylim([0, 100])
    ax.bar(x + 0.2, ya, 0.4, align="center", color=COLOR_A, label="train set")
    ax.bar(x - 0.2, yb, 0.4, align="center", color=COLOR_B, label="test set")
    title: str = "Percentage of Set at Risk"
    ax.set_title(f"{res['name']}\n{title}")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(linestyle="dotted", linewidth=1)
    ax.legend(loc="best")
    plt.margins(y=0)
    plt.tight_layout()
    filename = os.path.join(path, "cat_frac.png")
    fig.savefig(filename, pad_inches=0, bbox_inches="tight")
    logger.debug("Saved categorical fraction plot: %s", filename)


def _infer_categorical(target: Target, feature_id: int, threshold: float) -> dict:
    """Return the training and test set risks of a categorical feature."""
    return {
        "name": target.features[feature_id]["name"],
        "train": _infer(target, feature_id, threshold, True),
        "test": _infer(target, feature_id, threshold, False),
    }


def _is_categorical(target: Target, feature_id: int) -> bool:
    """Return whether a feature is categorical.

    For simplicity, assumes integer datatypes are categorical.
    """
    encoding: str = target.features[feature_id]["encoding"]
    return encoding[:3] in ("str", "int") or encoding[:6] in ("onehot")


def _attack_brute_force(
    target: Target,
    features: list[int],
    n_cpu: int,
    attack_threshold: float = 0,
) -> list[dict]:
    """Perform a brute force attribute inference attack.

    Computes the target model confidence scores for every value in the list and
    makes an inference if there is a unique highest confidence score that
    exceeds attack_threshold.
    """
    logger.debug("Brute force attacking categorical features")
    args = [(target, feature_id, attack_threshold) for feature_id in features]
    with mp.Pool(processes=n_cpu) as pool:
        return pool.starmap(_infer_categorical, args)


def _get_bounds_risk_for_sample(  # pylint: disable=too-many-locals,too-many-arguments
    target_model: BaseEstimator,
    feat_id: int,
    feat_min: float,
    feat_max: float,
    sample: np.ndarray,
    c_min: float = 0,
    protection_limit: float = 0.1,
    feat_n: int = 100,
) -> bool:
    """Return whether a quantitative feature is at risk for the sample.

    Returns a bool based on conditions surrounding upper and lower bounds of
    guesses that would lead to the same model confidence.

    Parameters
    ----------
    target_model : BaseEstimator
        Trained target model.
    feat_id : int
        Index of missing feature.
    feat_min : float
        Minimum value of missing feature.
    feat_max : float
        Maximum value of missing feature.
    sample : np.ndarray
        Original known feature vector.
    c_min : float
        Defines the confidence threshold below which we say we don't care.
    protection_limit : float
        Lower [upper] bound on estimated value must not be
        above[below] lower[upper] bounds e.g. 10% of value.
    feat_n : int
        Number of attribute values to test per sample.

    Returns
    -------
    bool
        Whether the quantitative feature is at risk for the sample.
    """
    # attribute values to test - linearly sampled
    x_feat = np.linspace(feat_min, feat_max, feat_n, endpoint=True)
    # get known label
    label: int = int(target_model.predict(sample.reshape(1, -1))[0])
    # a matrix containing feature vector with linearly spaced target attribute
    x1 = np.repeat(sample.reshape(1, -1), feat_n, axis=0)
    x1[:, feat_id] = x_feat
    # get the target model confidences across the attribute range
    confidences = target_model.predict_proba(x1)
    scores = confidences[:, label]  # scores just for the model predicted label
    peak: float = np.max(scores)
    # find lowest and highest values with peak confidence
    lower_bound_index: int = 0
    while scores[lower_bound_index] < peak:
        lower_bound_index += 1
    upper_bound_index: int = feat_n - 1
    while scores[upper_bound_index] < peak:
        upper_bound_index -= 1
    # condition 1: confidence in prediction above some threshold
    # condition 2: confidence for true value == max_confidence
    # condition 3: lower bound above lower protection limit
    # condition 4: upper bound of estimate below upper protection limit
    actual_value = sample[feat_id]
    actual_probs = target_model.predict_proba(sample.reshape(1, -1))[0]
    lower_bound: float = x_feat[lower_bound_index]
    upper_bound: float = x_feat[upper_bound_index]
    return (
        peak > c_min
        and actual_probs[label] == peak
        and lower_bound >= (1 - protection_limit) * actual_value
        and upper_bound <= (1 + protection_limit) * actual_value
    )


def _get_bounds_risk_for_feature(
    target_model: BaseEstimator, feature_id: int, samples: np.ndarray
) -> float:
    """Return the average feature risk score over a set of samples."""
    feature_risk: int = 0
    n_samples: int = len(samples)
    feat_min: float = np.min(samples[:, feature_id])
    feat_max: float = np.max(samples[:, feature_id])
    for i in range(n_samples):
        sample = samples[i]
        risk = _get_bounds_risk_for_sample(
            target_model, feature_id, feat_min, feat_max, sample
        )
        if risk:  # pragma:no cover
            # can be seen working in examples
            # testing uses nursery with dummy cont. feature
            # which is not predictive
            feature_risk += 1
    return feature_risk / n_samples if n_samples > 0 else 0


def _get_bounds_risk(
    target_model: BaseEstimator,
    feature_name: str,
    feature_id: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> dict:
    """Return a dict containing the dataset risks of a quantitative feature."""
    return {
        "name": feature_name,
        "train": _get_bounds_risk_for_feature(target_model, feature_id, X_train),
        "test": _get_bounds_risk_for_feature(target_model, feature_id, X_test),
    }


def _get_bounds_risks(target: Target, features: list[int], n_cpu: int) -> list[dict]:
    """Compute the bounds risk for all specified features."""
    logger.debug("Computing bounds risk for all specified features")
    args = [
        (
            target.model,
            target.features[feature_id]["name"],
            feature_id,
            target.X_train,
            target.X_test,
        )
        for feature_id in features
    ]
    with mp.Pool(processes=n_cpu) as pool:
        return pool.starmap(_get_bounds_risk, args)


def _attribute_inference(target: Target, n_cpu: int) -> dict:
    """Execute attribute inference attacks on a target given a trained model."""
    # brute force attack categorical attributes using dataset unique values
    logger.info("Attacking dataset: %s", target.dataset_name)
    logger.info("Attacking categorical attributes...")
    feature_list: list[int] = []
    for feature in range(target.n_features):
        if _is_categorical(target, feature):
            feature_list.append(feature)
    results_a: list[dict] = _attack_brute_force(target, feature_list, n_cpu)
    # compute risk scores for quantitative attributes
    logger.info("Attacking quantitative attributes...")
    feature_list = []
    for feature in range(target.n_features):
        if not _is_categorical(target, feature):
            feature_list.append(feature)
    results_b: list[dict] = _get_bounds_risks(target, feature_list, n_cpu)
    # combine results into single object
    return {
        "name": target.dataset_name,
        "categorical": results_a,
        "quantitative": results_b,
    }
