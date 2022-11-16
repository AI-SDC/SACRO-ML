"""
Attribute inference attacks.
"""

from __future__ import annotations

import json
import logging

# import pickle
from typing import Any

import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
from fpdf import FPDF
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder

from aisdc.attacks import report
from aisdc.attacks.attack import Attack
from aisdc.attacks.dataset import Data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aia")

COLOR_A: str = "#86bf91"  # training set plot colour
COLOR_B: str = "steelblue"  # testing set plot colour


class AttributeAttackArgs:
    """Arguments for attribute inference."""

    def __init__(self, **kwargs):
        self.__dict__["report_name"] = None
        self.__dict__["n_cpu"] = max(1, mp.cpu_count() - 1)
        self.__dict__.update(kwargs)

    def __str__(self):
        return ",".join(
            [f"{str(key)}: {str(value)}" for key, value in self.__dict__.items()]
        )

    def set_param(self, key: str, value: Any) -> None:
        """Set a parameter"""
        self.__dict__[key] = value

    def get_args(self) -> dict:
        """Return arguments"""
        return self.__dict__


class AttributeAttack(Attack):
    """Class to wrap the attribute inference attack code."""

    def __init__(self, args: AttributeAttackArgs = AttributeAttackArgs()):
        self.attack_metrics: dict = {}
        self.metadata: dict = {}
        self.args = args

    def __str__(self):
        return "Attribute inference attack"

    def attack(self, dataset: Data, target_model: BaseEstimator) -> None:
        """Programmatic attack entry point

        To be used when code has access to data class and trained target model

        Parameters
        ----------
        dataset: attacks.dataset.Data
            dataset as a Data class object
        target_model: sklearn.base.BaseEstimator
            target model that inherits from an sklearn BaseEstimator
        """
        self.attack_metrics = _attribute_inference(
            target_model, dataset, self.args.n_cpu
        )

    def _construct_metadata(self) -> None:
        """Constructs the metadata object. Called by the reporting method."""
        self.metadata = {}
        self.metadata["experiment_details"] = {}
        self.metadata["experiment_details"].update(self.args.__dict__)
        self.metadata["attack"] = str(self)

    def make_report(self) -> dict:
        """Create the report.

        Creates the output report. If self.args.report_name is not None, it will also save the
        information in json and pdf formats.

        Returns
        -------

        output: dict
            Dictionary containing all attack output.
        """
        output = {}
        logger.info("Starting report, report_name = %s", self.args.report_name)
        output["attack_metrics"] = self.attack_metrics
        self._construct_metadata()
        output["metadata"] = self.metadata
        if self.args.report_name is not None:
            json_report = json.dumps(output, cls=report.NumpyArrayEncoder)
            with open(f"{self.args.report_name}.json", "w", encoding="utf-8") as f:
                f.write(json_report)
            logger.info("Wrote report to %s", f"{self.args.report_name}.json")

            pdf = create_aia_report(output)
            pdf.output(f"{self.args.report_name}.pdf", "F")
            logger.info("Wrote pdf report to %s", f"{self.args.report_name}.pdf")
        return output


def _unique_max(confidences: list[float], threshold: float) -> bool:
    """Returns whether there is a unique maximum confidence value above
    threshold."""
    if len(confidences) > 0:
        max_conf = np.max(confidences)
        if max_conf < threshold:
            return False
        unique, count = np.unique(confidences, return_counts=True)
        for (u, c) in zip(unique, count):
            if c == 1 and u == max_conf:
                return True
    return False


def _get_inference_data(  # pylint: disable=too-many-locals
    target_model: BaseEstimator, dataset: Data, feature_id: int, memberset: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns a dataset of each sample with the attributes to test."""
    attack_feature: dict = dataset.features[feature_id]
    indices: list[int] = attack_feature["indices"]
    unique = np.unique(dataset.x_orig[:, feature_id])
    n_unique: int = len(unique)
    if attack_feature["encoding"] == "onehot":
        onehot_enc = OneHotEncoder()
        values = onehot_enc.fit_transform(unique.reshape(-1, 1)).toarray()
    else:  # pragma: no cover
        # catch all, but can't be reached because this func only called via _infer
        # which is only called for categorical data
        values = unique
    # samples after encoding (e.g. one-hot)
    samples: np.ndarray = dataset.x_train
    # samples before encoding (e.g. str)
    samples_orig: np.ndarray = dataset.x_train_orig
    if not memberset:
        samples = dataset.x_test
        samples_orig = dataset.x_test_orig
    n_samples, x_dim = np.shape(samples)
    x_values = np.zeros((n_samples, n_unique, x_dim), dtype=np.float64)
    y_values = target_model.predict(samples)
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
    target_model: BaseEstimator,
    dataset: Data,
    feature_id: int,
    threshold: float,
    memberset: bool,
) -> tuple[int, int, float, int, int]:
    """
    For each possible missing value, compute the confidence scores and
    label with the target model; if the label matches the known target model
    label for the original sample, and the highest confidence score is unique,
    infer that attribute if the confidence score is greater than a threshold.
    """
    logger.debug("Commencing attack on feature %d set %d", feature_id, int(memberset))
    correct: int = 0  # number of correct inferences made
    total: int = 0  # total number of inferences made
    x_values, y_values, baseline = _get_inference_data(
        target_model, dataset, feature_id, memberset
    )
    n_unique: int = len(x_values[1])
    samples = dataset.x_train if memberset else dataset.x_test
    for i, x in enumerate(x_values):  # each sample to perform inference on
        # get model confidence scores for all possible values for the sample
        confidence = target_model.predict_proba(x)
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
    """Returns a string report of the categorical results."""
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
                # no examples with test dataset where this doesn't happen
                msg += f"Unable to make any inferences of the {tranche} set\n"
    return msg


def report_quantitative(results: dict) -> str:
    """Returns a string report of the quantitative results."""
    results = results["quantitative"]
    msg = ""
    for feature in results:
        msg += (
            f"{feature['name']}: "
            f"{feature['train']:.2f} train risk, "
            f"{feature['test']:.2f} test risk\n"
        )
    return msg


def plot_quantitative_risk(res: dict, savefile: str = "") -> None:
    """Generates bar chart showing quantitative value risk scores."""
    logger.debug("Plotting quantitative feature risk scores")
    results = res["quantitative"]
    if len(results) < 1:  # pragma: no cover
        return
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
    if savefile != "":
        fig.savefig(savefile + "_quant_risk.png", pad_inches=0, bbox_inches="tight")
        logger.debug("Saved quantitative risk plot: %s", savefile)
    else:  # pragma: no cover
        plt.show()


def plot_categorical_risk(  # pylint: disable=too-many-locals
    res: dict, savefile: str = ""
) -> None:
    """Generates bar chart showing categorical risk scores."""
    logger.debug("Plotting categorical feature risk scores")
    results: list[dict] = res["categorical"]
    if len(results) < 1:  # pragma: no cover
        return
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
    if savefile != "":
        fig.savefig(savefile + "_cat_risk.png", pad_inches=0, bbox_inches="tight")
        logger.debug("Saved categorical risk plot: %s", savefile)
    else:  # pragma: no cover
        plt.show()


def plot_categorical_fraction(  # pylint: disable=too-many-locals
    res: dict, savefile: str = ""
) -> None:
    """Generates bar chart showing fraction of dataset inferred."""
    logger.debug("Plotting categorical feature tranche sizes")
    results: list[dict] = res["categorical"]
    if len(results) < 1:  # pragma: no cover
        return
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
    if savefile != "":
        fig.savefig(savefile + "_cat_frac.png", pad_inches=0, bbox_inches="tight")
        logger.debug("Saved categorical fraction plot: %s", savefile)
    else:  # pragma: no cover
        plt.show()


# def plot_from_file(filename: str, savefile: str = "") -> None: #pragma: no cover
#     """Loads a results save file and plots risk scores.
#        Has been tested but not iuncluded in unit tests or coverage
#        at this stage
#     """
#     logger.debug("Loading from results file: %s", filename)
#     with open(filename + ".pickle", "rb") as handle:
#         results = pickle.load(handle)
#     plot_categorical_risk(results, savefile=savefile)
#     plot_categorical_fraction(results, savefile=savefile)
#     plot_quantitative_risk(results, savefile=savefile)


def _infer_categorical(
    target_model: BaseEstimator, dataset: Data, feature_id: int, threshold: float
) -> dict:
    """Returns the training and test set risks of a categorical feature."""
    result: dict = {
        "name": dataset.features[feature_id]["name"],
        "train": _infer(target_model, dataset, feature_id, threshold, True),
        "test": _infer(target_model, dataset, feature_id, threshold, False),
    }
    return result


def _is_categorical(dataset: Data, feature_id: int) -> bool:
    """Returns whether a feature is categorical.
    For simplicity, assumes integer datatypes are categorical."""
    encoding: str = dataset.features[feature_id]["encoding"]
    if encoding[:3] in ("str", "int") or encoding[:6] in ("onehot"):
        return True
    return False


def _attack_brute_force(
    target_model: BaseEstimator,
    dataset: Data,
    features: list[int],
    n_cpu: int,
    attack_threshold: float = 0,
) -> list[dict]:
    """
    Performs a brute force attribute inference attack by computing the target
    model confidence scores for every value in the list and making an inference
    if there is a unique highest confidence score that exceeds attack_threshold.
    """
    logger.debug("Brute force attacking categorical features")
    args = [
        (target_model, dataset, feature_id, attack_threshold) for feature_id in features
    ]
    with mp.Pool(processes=n_cpu) as pool:  # pylint:disable=not-callable
        results = pool.starmap(_infer_categorical, args)
    return results


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
    """Returns a bool based on conditions surrounding upper and lower bounds of
    guesses that would lead to the same model confidence.

    Parameters
    ----------

    target_model: BaseEstimator
        Trained target model.
    feat_id: int
        Index of missing feature.
    feat_min: float
        Minimum value of missing feature.
    feat_max: float
        Maximum value of missing feature.
    sample: np.ndarray
        Original known feature vector.
    c_min: float
        Defines the confidence threshold below which we say we don't care.
    protection_limit: float
        Lower [upper] bound on estimated value must not be
        above[below] lower[upper] bounds e.g. 10% of value.
    feat_n: int
        Number of attribute values to test per sample.

    Returns
    -------
    A bool representing whether the quantitative feature is at risk for the sample.

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
    if (
        peak > c_min
        and actual_probs[label] == peak
        and lower_bound >= (1 - protection_limit) * actual_value
        and upper_bound <= (1 + protection_limit) * actual_value
    ):  # pragma: no cover
        # has been tested and shown to work in examples notebooks
        # but tests use simplistic nursery dataset
        return True
    return False


def _get_bounds_risk_for_feature(
    target_model: BaseEstimator, feature_id: int, samples: np.ndarray
) -> float:
    """Returns the average feature risk score over a set of samples."""
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
    if n_samples < 1:  # pragma: no cover
        # is unreachable because of how it is called
        return 0
    return feature_risk / n_samples


def _get_bounds_risk(
    target_model: BaseEstimator,
    feature_name: str,
    feature_id: int,
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> dict:
    """Returns a dictionary containing the training and test set risks of a
    quantitative feature."""
    risk: dict = {
        "name": feature_name,
        "train": _get_bounds_risk_for_feature(target_model, feature_id, x_train),
        "test": _get_bounds_risk_for_feature(target_model, feature_id, x_test),
    }
    return risk


def _get_bounds_risks(
    target_model: BaseEstimator, dataset: Data, features: list[int], n_cpu: int
) -> list[dict]:
    """Computes the bounds risk for all specified features."""
    logger.debug("Computing bounds risk for all specified features")
    args = [
        (
            target_model,
            dataset.features[feature_id]["name"],
            feature_id,
            dataset.x_train,
            dataset.x_test,
        )
        for feature_id in features
    ]
    with mp.Pool(processes=n_cpu) as pool:  # pylint:disable=not-callable
        results = pool.starmap(_get_bounds_risk, args)
    return results


def _attribute_inference(
    target_model: BaseEstimator,
    dataset: Data,
    n_cpu: int,
) -> dict:
    """
    Execute attribute inference attacks on a dataset given a trained model.
    """
    # brute force attack categorical attributes using dataset unique values
    logger.debug("Attacking dataset: %s", dataset.name)
    logger.debug("Attacking categorical attributes...")
    feature_list: list[int] = []
    for feature in range(dataset.n_features):
        if _is_categorical(dataset, feature):
            feature_list.append(feature)
    results_a: list[dict] = _attack_brute_force(
        target_model, dataset, feature_list, n_cpu
    )
    # compute risk scores for quantitative attributes
    logger.debug("Attacking quantitative attributes...")
    feature_list = []
    for feature in range(dataset.n_features):
        if not _is_categorical(dataset, feature):
            feature_list.append(feature)
    results_b: list[dict] = _get_bounds_risks(
        target_model, dataset, feature_list, n_cpu
    )
    # combine results into single object
    results: dict = {
        "name": dataset.name,
        "categorical": results_a,
        "quantitative": results_b,
    }
    return results


def create_aia_report(output: dict, name: str = "aia_report") -> FPDF:
    """Creates PDF report."""
    aia_metrics = output["attack_metrics"]
    metadata = output["metadata"]
    plot_categorical_risk(aia_metrics, name)
    plot_categorical_fraction(aia_metrics, name)
    plot_quantitative_risk(aia_metrics, name)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_xy(0, 0)
    report.title(pdf, "Attribute Inference Attack Report")
    report.subtitle(pdf, "Introduction")
    report.subtitle(pdf, "Metadata")
    for key, value in metadata["experiment_details"].items():
        report.line(pdf, f"{key:>30s}: {str(value):30s}", font="courier")
    report.subtitle(pdf, "Metrics")
    categ_rep = report_categorical(aia_metrics).split("\n")
    quant_rep = report_quantitative(aia_metrics).split("\n")
    report.line(pdf, "Categorical Features:", font="courier")
    for line in categ_rep:
        report.line(pdf, line, font="courier")
    report.line(pdf, "Quantitative Features:", font="courier")
    for line in quant_rep:
        report.line(pdf, line, font="courier")
    pdf.add_page()
    report.subtitle(pdf, "Plots")
    if len(aia_metrics["categorical"]) > 0:
        pdf.image(name + "_cat_risk.png", x=None, y=None, w=150, h=0, type="", link="")
        pdf.image(name + "_cat_frac.png", x=None, y=None, w=150, h=0, type="", link="")
    if len(aia_metrics["quantitative"]) > 0:
        pdf.image(
            name + "_quant_risk.png", x=None, y=None, w=150, h=0, type="", link=""
        )
    return pdf
