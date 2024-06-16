"""Generate report for TREs from JSON file."""

from __future__ import annotations

import json
import os
import pathlib
import pprint
import shutil
from datetime import date

import matplotlib.pyplot as plt
import numpy as np


def cleanup_files_for_release(
    move_into_artefacts: list[str],
    copy_into_release: list[str],
    release_dir: str = "release_files",
    artefacts_dir: str = "training_artefacts",
) -> None:
    """Move files created during the release process into appropriate folders."""
    if not os.path.exists(release_dir):
        os.makedirs(release_dir)

    if not os.path.exists(artefacts_dir):
        os.makedirs(artefacts_dir)

    for filepath in move_into_artefacts:
        if os.path.exists(str(filepath)):
            filename = os.path.basename(filepath)
            dest = os.path.join(artefacts_dir, filename)
            shutil.move(filename, dest)

    for filepath in copy_into_release:
        if os.path.exists(str(filepath)):
            filename = os.path.basename(filepath)
            dest = os.path.join(release_dir, filename)
            shutil.copy(filepath, dest)


class GenerateJSONModule:
    """Create and append to a JSON file."""

    def __init__(self, filename: str | None = None) -> None:
        self.filename = filename

        if self.filename is None:
            self.filename = (
                "ATTACK_RESULTS" + str(date.today().strftime("%d_%m_%Y")) + ".json"
            )

        dirname = os.path.normpath(os.path.dirname(self.filename))
        os.makedirs(dirname, exist_ok=True)
        # if file doesn't exist, create it
        if not os.path.exists(self.filename):
            with open(self.filename, "w", encoding="utf-8") as f:
                f.write("")

    def add_attack_output(self, incoming_json: dict, class_name: str) -> None:
        """Add a section of JSON to the file which is already open."""
        # Read the contents of the file and then clear the file
        with open(self.filename, "r+", encoding="utf-8") as f:
            file_contents = f.read()
            file_data = json.loads(file_contents) if file_contents != "" else {}
            f.truncate(0)

        # Add the new JSON to the JSON that was in the file, and re-write
        with open(self.filename, "w", encoding="utf-8") as f:
            incoming_json = json.loads(incoming_json)

            if "log_id" in incoming_json:
                class_name = class_name + "_" + str(incoming_json["log_id"])

            file_data[class_name] = incoming_json
            json.dump(file_data, f)

    def get_output_filename(self) -> str:
        """Return the filename of the JSON file which has been created."""
        return self.filename

    def clean_file(self) -> None:
        """Delete the file if it exists."""
        if os.path.exists(self.filename):
            os.remove(self.filename)

        with open(self.filename, "w", encoding="utf-8") as f:
            f.write("")


class AnalysisModule:
    """Wrapper module for metrics analysis modules."""

    def __init__(self) -> None:
        self.immediate_rejection = []
        self.support_rejection = []
        self.support_release = []

    def process_dict(self) -> dict:
        """Produce a risk summary output based on analysis in this module."""
        raise NotImplementedError()

    def get_recommendation(self) -> tuple:
        """Return the three recommendation buckets created by this module."""
        return self.immediate_rejection, self.support_rejection, self.support_release

    def __str__(self) -> str:
        """Return the string representation of an analysis module."""
        raise NotImplementedError()


class FinalRecommendationModule(AnalysisModule):  # pylint: disable=too-many-instance-attributes
    """Generate the first layer of a recommendation report."""

    def __init__(self, report: dict) -> None:
        super().__init__()

        self.P_VAL_THRESH = 0.05
        self.MEAN_AUC_THRESH = 0.65
        self.INSTANCE_MODEL_WEIGHTING_SCORE = 5
        self.MIN_SAMPLES_LEAF_SCORE = 5
        self.STATISTICALLY_SIGNIFICANT_SCORE = 2
        self.MEAN_AUC_SCORE = 4

        self.report = report
        self.scores = []
        self.reasons = []

    def _is_instance_based_model(self, instance_based_model_score) -> bool:
        if "model_name" in self.report:
            if self.report["model_name"] == "SVC":
                self.scores.append(instance_based_model_score)
                self.reasons.append("Model is SVM (supersedes all other tests)")
                self.immediate_rejection.append("Model is SVM")
                return True
            if self.report["model_name"] == "KNeighborsClassifier":
                self.scores.append(instance_based_model_score)
                self.reasons.append("Model is kNN (supersedes all other tests)")
                self.immediate_rejection.append("Model is kNN")
                return True
        return False

    def _tree_min_samples_leaf(self, min_samples_leaf_score: int | float) -> None:
        # Find min samples per leaf requirement
        base_path = pathlib.Path(__file__).parents[1]
        risk_appetite_path = os.path.join(base_path, "safemodel", "rules.json")
        min_samples_leaf_appetite = None

        with open(risk_appetite_path, "r+", encoding="utf-8") as f:
            file_contents = f.read()
            json_structure = json.loads(file_contents)

            rules = json_structure["DecisionTreeClassifier"]["rules"]
            for entry in rules:
                if (
                    "keyword" in entry
                    and entry["keyword"] == "min_samples_leaf"
                    and "operator" in entry
                    and entry["operator"] == "min"
                ):
                    min_samples_leaf_appetite = entry["value"]
                    break

        if (
            ("model_params" in self.report)
            and min_samples_leaf_appetite is not None
            and "min_samples_leaf" in self.report["model_params"]
        ):
            min_samples_leaf = self.report["model_params"]["min_samples_leaf"]
            if min_samples_leaf < min_samples_leaf_appetite:
                self.scores.append(min_samples_leaf_score)

                msg = "Min samples per leaf < " + str(min_samples_leaf_appetite)
                self.reasons.append(msg)
                self.support_rejection.append(msg)
            else:
                msg = "Min samples per leaf > " + str(min_samples_leaf_appetite)
                self.support_release.append(msg)

    def _statistically_significant_auc(
        self,
        p_val_thresh: float,
        mean_auc_thresh: float,
        stat_sig_score: float,
        mean_auc_score: float,
    ) -> None:
        stat_sig_auc = []
        for k in self.report:
            if (
                isinstance(self.report[k], dict)
                and "attack_experiment_logger" in self.report[k]
            ):
                for i in self.report[k]["attack_experiment_logger"][
                    "attack_instance_logger"
                ]:
                    instance = self.report[k]["attack_experiment_logger"][
                        "attack_instance_logger"
                    ][i]

                    auc_key = "P_HIGHER_AUC"
                    if auc_key in instance and instance[auc_key] < p_val_thresh:
                        stat_sig_auc.append(instance["AUC"])

                n_instances = len(
                    self.report[k]["attack_experiment_logger"]["attack_instance_logger"]
                )
                if len(stat_sig_auc) / n_instances > 0.1:
                    msg = ">10% AUC are statistically significant in experiment " + str(
                        k
                    )

                    self.scores.append(stat_sig_score)
                    self.reasons.append(msg)
                    self.support_rejection.append(msg)
                else:
                    msg = "<10% AUC are statistically significant in experiment " + str(
                        k
                    )
                    self.support_release.append(msg)

                if len(stat_sig_auc) > 0:
                    mean = np.mean(np.array(stat_sig_auc))
                    if mean > mean_auc_thresh:
                        msg = "Attack AUC > threshold of " + str(mean_auc_thresh)
                        msg = msg + " in experiment " + str(k)

                        self.scores.append(mean_auc_score)
                        self.reasons.append(msg)
                        self.support_rejection.append(msg)
                    else:
                        msg = "Attack AUC <= threshold of " + str(mean_auc_thresh)
                        msg = msg + " in experiment " + str(k)
                        self.support_release.append(msg)

    def process_dict(self) -> dict:
        """Return a dictionary summarising the metrics."""
        self._tree_min_samples_leaf(self.MIN_SAMPLES_LEAF_SCORE)
        self._statistically_significant_auc(
            self.P_VAL_THRESH,
            self.MEAN_AUC_THRESH,
            self.STATISTICALLY_SIGNIFICANT_SCORE,
            self.MEAN_AUC_SCORE,
        )

        if len(self.scores) == 0:
            summarised_score = 0
        else:
            summarised_score = int(np.sum(np.array(self.scores)).round(0))
            summarised_score = min(summarised_score, 5)

        # if model is instance based, it is automatically disclosive. Assign max score
        if self._is_instance_based_model(self.INSTANCE_MODEL_WEIGHTING_SCORE):
            summarised_score = self.INSTANCE_MODEL_WEIGHTING_SCORE

        return {}

    def __str__(self) -> str:
        """Return string representation of the final recommendation."""
        return "Final Recommendation"


class SummariseUnivariateMetricsModule(AnalysisModule):
    """Summarise a set of chosen univariate metrics from the output dictionary."""

    def __init__(self, report: dict, metrics_list=None) -> None:
        super().__init__()

        if metrics_list is None:
            metrics_list = ["AUC", "ACC", "FDIF01"]

        self.report = report
        self.metrics_list = metrics_list

    def process_dict(self) -> dict:
        """Return a dictionary summarising the metrics."""
        output_dict = {}

        for k in self.report:
            if (
                isinstance(self.report[k], dict)
                and "attack_experiment_logger" in self.report[k]
            ):
                metrics_dict = {m: [] for m in self.metrics_list}
                for _, iteration_value in self.report[k]["attack_experiment_logger"][
                    "attack_instance_logger"
                ].items():
                    for m in metrics_dict:
                        metrics_dict[m].append(iteration_value[m])
                output = {}
                for m in self.metrics_list:
                    output[m] = {
                        "min": min(metrics_dict[m]),
                        "max": max(metrics_dict[m]),
                        "mean": np.mean(metrics_dict[m]),
                        "median": np.median(metrics_dict[m]),
                    }
                output_dict[k] = output
        return output_dict

    def __str__(self) -> str:
        """Return the string representation of a univariate metrics module."""
        return "Summary of Univarite Metrics"


class SummariseAUCPvalsModule(AnalysisModule):
    """Summarise a list of AUC values."""

    def __init__(
        self, report: dict, p_thresh: float = 0.05, correction: str = "bh"
    ) -> None:
        super().__init__()

        self.report = report
        self.p_thresh = p_thresh
        self.correction = correction

    def _n_sig(self, p_val_list: list[float], correction: str = "none") -> int:
        """Compute the number of significant p-vals with different corrections."""
        if correction == "none":
            return len(np.where(np.array(p_val_list) <= self.p_thresh)[0])
        if correction == "bh":
            sorted_p = sorted(p_val_list)
            m = len(p_val_list)
            alpha = self.p_thresh
            comparators = np.arange(1, m + 1, 1) * alpha / m
            return (sorted_p <= comparators).sum()
        if correction == "bo":  # bonferroni
            return len(
                np.where(np.array(p_val_list) <= self.p_thresh / len(p_val_list))[0]
            )

        raise NotImplementedError()  # any others?

    def _get_metrics_list(self) -> list[float]:
        metrics_list = []
        for k in self.report:
            if (
                isinstance(self.report[k], dict)
                and "attack_experiment_logger" in self.report[k]
            ):
                for _, iteration_value in self.report[k]["attack_experiment_logger"][
                    "attack_instance_logger"
                ].items():
                    metrics_list.append(iteration_value["P_HIGHER_AUC"])
        return metrics_list

    def process_dict(self) -> dict:
        """Process the dict to summarise the number of significant AUC p-values."""
        p_val_list = self._get_metrics_list()
        return {
            "n_total": len(p_val_list),
            "p_thresh": self.p_thresh,
            "n_sig_uncorrected": self._n_sig(p_val_list),
            "correction": self.correction,
            "n_sig_corrected": self._n_sig(p_val_list, self.correction),
        }

    def __str__(self) -> str:
        """Return the string representation of a AUC p-values module."""
        return f"Summary of AUC p-values at p = ({self.p_thresh})"


class SummariseFDIFPvalsModule(SummariseAUCPvalsModule):
    """Summarise the number of significant FDIF p-values."""

    def get_metric_list(self, input_dict: dict) -> list[float]:
        """Get metrics_list from attack_instance_logger within JSON file."""
        metric_list = []
        for _, iteration_value in input_dict["attack_instance_logger"].items():
            metric_list.append(iteration_value["PDIF01"])
        return [np.exp(-m) for m in metric_list]

    def __str__(self) -> str:
        """Return the string representation of a FDIF p-values module."""
        return f"Summary of FDIF p-values at p = ({self.p_thresh})"


class LogLogROCModule(AnalysisModule):
    """Generate a log-log plot."""

    def __init__(
        self, report: dict, output_folder: str | None = None, include_mean: bool = True
    ) -> None:
        super().__init__()

        self.report = report
        self.output_folder = output_folder
        self.include_mean = include_mean

    def process_dict(self) -> str:
        """Create a roc plot for multiple repetitions."""
        log_plot_names = []

        for k in self.report:
            if (
                isinstance(self.report[k], dict)
                and "attack_experiment_logger" in self.report[k]
            ):
                plt.figure(figsize=(8, 8))
                plt.plot([0, 1], [0, 1], "k--")

                # Compute average ROC
                base_fpr = np.linspace(0, 1, 1000)
                metrics = self.report[k]["attack_experiment_logger"][
                    "attack_instance_logger"
                ].values()
                all_tpr = np.zeros((len(metrics), len(base_fpr)), float)

                for i, metric_set in enumerate(metrics):
                    all_tpr[i, :] = np.interp(
                        base_fpr, metric_set["fpr"], metric_set["tpr"]
                    )

                for _, metric_set in enumerate(metrics):
                    plt.plot(
                        metric_set["fpr"],
                        metric_set["tpr"],
                        color="lightsalmon",
                        linewidth=0.5,
                    )

                tpr_mu = all_tpr.mean(axis=0)
                plt.plot(base_fpr, tpr_mu, "r")

                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.xscale("log")
                plt.yscale("log")
                plt.tight_layout()
                plt.grid()
                out_file = (
                    f"{self.report['log_id']}-{self.report['metadata']['attack']}.png"
                )
                if self.output_folder is not None:
                    out_file = os.path.join(self.output_folder, out_file)
                plt.savefig(out_file)
                log_plot_names.append(out_file)
        return "Log plot(s) saved to " + str(log_plot_names)

    def __str__(self) -> str:
        """Return the string representation of a ROC log plot module."""
        return "ROC Log Plot"


class GenerateTextReport:
    """Generate a text report from a JSON input."""

    def __init__(self) -> None:
        self.text_out = []
        self.target_json_filename = None
        self.attack_json_filename = None
        self.model_name_from_target = None

        self.immediate_rejection = []
        self.support_rejection = []
        self.support_release = []

    def _process_target_json(self) -> None:
        """Create a summary of a target model JSON file."""
        model_params_of_interest = [
            "C",
            "kernel",
            "n_neighbors",
            "hidden_layer_sizes",
            "activation",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "n_estimators",
            "learning_rate",
        ]

        with open(self.target_json_filename, encoding="utf-8") as f:
            json_report = json.loads(f.read())

        output_string = "TARGET MODEL SUMMARY\n"

        if "model_name" in json_report:
            output_string = (
                output_string + "model_name: " + json_report["model_name"] + "\n"
            )

        if "n_samples" in json_report:
            output_string = output_string + "number of samples used to train: "
            output_string = output_string + str(json_report["n_samples"]) + "\n"

        if "model_params" in json_report:
            for param in model_params_of_interest:
                if param in json_report["model_params"]:
                    output_string = output_string + param + ": "
                    output_string = output_string + str(
                        json_report["model_params"][param]
                    )
                    output_string = output_string + "\n"

        if "model_path" in json_report:
            filepath = os.path.split(os.path.abspath(self.target_json_filename))[0]
            self.model_name_from_target = os.path.join(
                filepath, json_report["model_path"]
            )

        self.text_out.append(output_string)

    def pretty_print(self, report: dict, title: str) -> str:
        """Format JSON code to make it more readable for TREs."""
        returned_string = title + "\n"
        for key in report:
            returned_string = returned_string + key + "\n"
            returned_string = returned_string + pprint.pformat(report[key]) + "\n\n"
        return returned_string

    def process_attack_target_json(
        self, attack_filename: str, target_filename: str = None
    ) -> None:
        """Create a neat summary of an attack JSON file."""
        self.attack_json_filename = attack_filename

        with open(attack_filename, encoding="utf-8") as f:
            json_report = json.loads(f.read())

        if target_filename is not None:
            self.target_json_filename = target_filename

            with open(target_filename, encoding="utf-8") as f:
                target_file = json.loads(f.read())
                json_report = {**json_report, **target_file}

        modules = [
            FinalRecommendationModule(json_report),
        ]

        for m in modules:
            output = m.process_dict()
            returned = m.get_recommendation()

            self.immediate_rejection += returned[0]
            self.support_rejection += returned[1]
            self.support_release += returned[2]

        output_string = self.pretty_print(output, "ATTACK JSON RESULTS")

        self.text_out.append(output_string)

        bucket_text = "Immediate rejection recommended for the following reason:\n"
        if len(self.immediate_rejection) > 0:
            for reason in self.immediate_rejection:
                bucket_text += str(reason) + "\n"
        else:
            bucket_text += "None\n"

        bucket_text += "\nEvidence supporting rejection:\n"
        if len(self.support_rejection) > 0:
            for reason in self.support_rejection:
                bucket_text += str(reason) + "\n"
        else:
            bucket_text += "None\n"

        bucket_text += "\nEvidence supporting release:\n"
        if len(self.support_release) > 0:
            for reason in self.support_release:
                bucket_text += str(reason) + "\n"
        else:
            bucket_text += "None\n"

        self.text_out.append(bucket_text)

    def export_to_file(  # pylint: disable=too-many-arguments
        self,
        output_filename: str = "summary.txt",
        move_files: bool = False,
        model_filename: str | None = None,
        release_dir: str = "release_files",
        artefacts_dir: str = "training_artefacts",
    ) -> None:
        """Take the input strings collected and combine into a neat text file."""
        copy_of_text_out = self.text_out
        self.text_out = []

        if self.target_json_filename is not None:
            self._process_target_json()

        self.text_out += copy_of_text_out

        output_filename = output_filename.replace(" ", "_")

        with open(output_filename, "w", encoding="utf-8") as text_file:
            for output_string in self.text_out:
                text_file.write(output_string)
                text_file.write("\n")

        if move_files is True:
            move_into_artefacts = ["log_roc.png"]

            copy_into_release = [
                output_filename,
                self.attack_json_filename,
                self.target_json_filename,
            ]

            if model_filename is None:
                copy_into_release.append(self.model_name_from_target)
            else:
                copy_into_release.append(model_filename)

            cleanup_files_for_release(
                move_into_artefacts, copy_into_release, release_dir, artefacts_dir
            )
