"""
Generate report for TREs from JSON file
"""

import json
import os
import pprint

import matplotlib.pyplot as plt
import numpy as np


class AnalysisModule:
    """
    Wrapper module for metrics analysis modules
    """

    def process_dict(self):
        """
        Function that produces a risk summary output based on analysis in this module
        """
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()


class FinalRecommendationModule(
    AnalysisModule
):  # pylint: disable=too-many-instance-attributes
    """
    Module that generates the first layer of a recommendation report
    """

    def __init__(self, report: dict):
        self.P_VAL_THRESH = 0.05
        self.MEAN_AUC_THRESH = 0.65

        self.SVM_WEIGHTING_SCORE = 5
        self.MIN_SAMPLES_LEAF_SCORE = 3
        self.STATISTICALLY_SIGNIFICANT_SCORE = 2
        self.MEAN_AUC_SCORE = 4

        self.report = report

        self.scores = []
        self.reasons = []

    def _is_svm(self, svm_weighing_score):
        if "model" in self.report:
            if self.report["model"] == "SVC":
                self.scores.append(svm_weighing_score)
                self.reasons.append("Model is SVM (supersedes all other tests)")
                return True
        return False

    def _rf_min_samples_leaf(self, min_samples_leaf_score):
        if "model_params" in self.report:
            if "min_samples_leaf" in self.report["model_params"]:
                min_samples_leaf = self.report["model_params"]["min_samples_leaf"]
                if min_samples_leaf < 5:
                    self.scores.append(min_samples_leaf_score)
                    self.reasons.append("Min samples per leaf < 5")

    def _statistically_significant_auc(
        self, p_val_thresh, mean_auc_thresh, stat_sig_score, mean_auc_score
    ):
        stat_sig_auc = []
        if "attack_experiment_logger" in self.report:
            for i in self.report["attack_experiment_logger"]["attack_instance_logger"]:
                instance = self.report["attack_experiment_logger"][
                    "attack_instance_logger"
                ][i]
                if instance["P_HIGHER_AUC"] < p_val_thresh:
                    stat_sig_auc.append(instance["AUC"])

            n_instances = len(
                self.report["attack_experiment_logger"]["attack_instance_logger"]
            )
            if (
                len(stat_sig_auc) / n_instances > 0.1
            ):  # > 10% of AUC are statistically significant
                self.scores.append(stat_sig_score)
                self.reasons.append(">10% AUC are statistically significant")

            if len(stat_sig_auc) > 0:
                mean = np.mean(np.array(stat_sig_auc))
                if mean > mean_auc_thresh:
                    self.scores.append(mean_auc_score)
                    self.reasons.append("Attack AUC > threshold")

    def process_dict(self) -> dict:
        self._rf_min_samples_leaf(self.MIN_SAMPLES_LEAF_SCORE)
        self._statistically_significant_auc(
            self.P_VAL_THRESH,
            self.MEAN_AUC_THRESH,
            self.STATISTICALLY_SIGNIFICANT_SCORE,
            self.MEAN_AUC_SCORE,
        )

        if len(self.scores) == 0:
            summarised_score = 0
        else:
            summarised_score = int(np.mean(np.array(self.scores)).round(0))

        # if model is instance based, it is automatically disclosive. Assign max score
        if self._is_svm(self.SVM_WEIGHTING_SCORE):
            summarised_score = self.SVM_WEIGHTING_SCORE

        output = dict()
        output["final_score"] = summarised_score
        output["score_breakdown"] = self.scores
        output["score_descriptions"] = self.reasons

        return output

    def __str__(self):
        return "Final Recommendation"


class SummariseUnivariateMetricsModule(AnalysisModule):
    """
    Module that summarises a set of chosen univariate metrics from the output dictionary
    """

    def __init__(self, report: dict, metrics_list=None):
        if metrics_list is None:
            metrics_list = ["AUC", "ACC", "FDIF01"]

        self.report = report
        self.metrics_list = metrics_list

    def process_dict(self) -> dict:
        metrics_dict = {m: [] for m in self.metrics_list}
        for _, iteration_value in self.report["attack_experiment_logger"][
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
        return output

    def __str__(self):
        return "Summary of Univarite Metrics"


class SummariseAUCPvalsModule(AnalysisModule):
    """
    Module that summarises a list of AUC values
    """

    def __init__(self, report: dict, p_thresh: float = 0.05, correction: str = "bh"):
        self.report = report
        self.p_thresh = p_thresh
        self.correction = correction

    def _n_sig(self, p_val_list: list[float], correction: str = "none") -> int:
        """Compute the number of significant p-vals in a list with different corrections for
        multiple testing"""

        if correction == "none":
            return len(np.where(np.array(p_val_list) <= self.p_thresh)[0])
        elif correction == "bh":
            sorted_p = sorted(p_val_list)
            m = len(p_val_list)
            alpha = self.p_thresh
            comparators = np.arange(1, m + 1, 1) * alpha / m
            return (sorted_p <= comparators).sum()
        elif correction == "bo":  # bonferroni
            return len(
                np.where(np.array(p_val_list) <= self.p_thresh / len(p_val_list))[0]
            )
        else:
            raise NotImplementedError()  # any others?

    def _get_metrics_list(self) -> list[float]:
        metrics_list = []
        for _, iteration_value in self.report["attack_experiment_logger"][
            "attack_instance_logger"
        ].items():
            metrics_list.append(iteration_value["P_HIGHER_AUC"])
        return metrics_list

    def process_dict(self) -> dict:
        """Process the dict to summarise the number of significant AUC p-values"""
        p_val_list = self._get_metrics_list()
        output = {
            "n_total": len(p_val_list),
            "p_thresh": self.p_thresh,
            "n_sig_uncorrected": self._n_sig(p_val_list),
            "correction": self.correction,
            "n_sig_corrected": self._n_sig(p_val_list, self.correction),
        }
        return output

    def __str__(self):
        return f"Summary of AUC p-values at p = ({self.p_thresh})"


class SummariseFDIFPvalsModule(SummariseAUCPvalsModule):
    """Summarise the number of significant FDIF p-values"""

    # TODO do we want to parameterise which FDIF (01, 001, etc)?
    def get_metric_list(self, input_dict: dict) -> list[float]:
        metric_list = []
        for _, iteration_value in input_dict["attack_instance_logger"].items():
            metric_list.append(iteration_value["PDIF01"])
        metric_list = [np.exp(-m) for m in metric_list]
        return metric_list

    def __str__(self):
        return f"Summary of FDIF p-values at p = ({self.p_thresh})"


class LogLogROCModule(AnalysisModule):
    """
    Module that generates a log-log plot
    """

    def __init__(self, report: dict, output_folder=None, include_mean=True):
        self.report = report
        self.output_folder = output_folder
        self.include_mean = include_mean

    def process_dict(self):
        """Create a roc plot for multiple repetitions"""
        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], "k--")

        # Compute average ROC
        base_fpr = np.linspace(0, 1, 1000)
        metrics = self.report["attack_experiment_logger"][
            "attack_instance_logger"
        ].values()
        all_tpr = np.zeros((len(metrics), len(base_fpr)), float)

        for i, metric_set in enumerate(metrics):
            all_tpr[i, :] = np.interp(base_fpr, metric_set["fpr"], metric_set["tpr"])

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
        output_file_name = (
            f"{self.report['log_id']}-{self.report['metadata']['attack']}.png"
        )
        if self.output_folder is not None:
            output_file_name = os.path.join(self.output_folder, output_file_name)
        plt.savefig(output_file_name)
        return "Log plot saved to " + output_file_name

    def __str__(self):
        return "ROC Log Plot"


def pretty_print(report: dict) -> str:
    """
    Function that formats JSON code to make it more readable for TREs
    """
    returned_string = ""

    for key in report.keys():
        returned_string = returned_string + key + "\n"
        returned_string = returned_string + pprint.pformat(report[key]) + "\n\n"

    return returned_string


def process_json(input_filename: str, output_filename: str):
    """
    Function that takes an input JSON filename and outputs a neat text file summarising results
    """
    with open(input_filename) as f:
        json_report = json.loads(f.read())

    modules = [
        FinalRecommendationModule(json_report),
    ]

    output = {str(m): m.process_dict() for m in modules}
    output_string = pretty_print(output)

    with open(output_filename, "w") as text_file:
        text_file.write(output_string)
