"""Code for automatic report generation."""

import json
import os
from typing import Any

import numpy as np
import pylab as plt
from fpdf import FPDF
from pypdf import PdfWriter

from sacroml.attacks.attack_report_formatter import GenerateJSONModule

# Adds a border to all pdf cells of set to 1 -- useful for debugging
BORDER = 0

DISPLAY_METRICS = [
    "AUC",
    "ACC",
    "Advantage",
    "FDIF01",
    "PDIF01",
    "TPR@0.1",
    "TPR@0.01",
    "TPR@0.001",
    "TPR@1e-05",
]

MAPPINGS = {"PDIF01": lambda x: np.exp(-x)}

INTRODUCTION = (
    "This report provides a summary of a series of simulated attack experiments "
    "performed on the model outputs provided. An attack model is trained to "
    "attempt to distinguish between outputs from training (in-sample) and "
    "testing (out-of-sample) data. The metrics below describe the success of "
    "this classifier. A successful classifier indicates that the original model "
    "is unsafe and should not be allowed to be released from the TRE.\n In "
    "particular, the simulation splits the data provided into test and train "
    "sets (each will in- and out-of-sample examples). The classifier is trained "
    "on the train set and evaluated on the test set. This is repeated with "
    "different train/test splits a user-specified number of times.\n To help "
    "place the results in context, the code may also have run a series of "
    "baseline experiments. In these, random model outputs for hypothetical in- "
    "and out-of-sample data are generated with identical statistical properties. "
    "In these baseline cases, there is no signal that an attacker could leverage "
    "and therefore these values provide a baseline against which the actual "
    "values can be compared.\n For some metrics (FDIF and AUC), we are able to "
    "compute p-values. In each case, shown below (in the Global metrics "
    "sections) is the number of repetitions that exceeded the p-value threshold "
    "both without, and with correction for multiple testing (Benjamini-Hochberg "
    "procedure).\n ROC curves for all real (red) and dummy (blue) repetitions are "
    "provided. These are shown in log space (as reommended here [ADD URL]) to "
    "emphasise the region in which risk is highest -- the bottom left (are high "
    "true positive rates possible with low false positive rates).\n A description "
    "of the metrics and how to interpret them within the context of an attack is "
    "given below."
)

LOGROC_CAPTION = (
    "This plot shows the False Positive Rate (x) versus the True Positive Rate "
    "(y). The axes are in log space enabling us to focus on areas where the "
    "False Positive Rate is low (left hand area). Curves above the y = x line "
    "(black dashes) in this region represent a disclosure risk as an attacker "
    "can obtain many more true than false positives. The solid coloured lines "
    "show the curves for the attack simulations with the true model outputs. The "
    "lighter grey lines show the curves for randomly generated outputs with no "
    "structure (i.e. in- and out-of- sample predictions are generated from the "
    "same distributions. Solid curves consistently higher than the "
    "grey curves in the left hand part of the plot are a sign of concern. "
)

GLOSSARY = {
    "AUC": "Area Under the ROC curve",
    "True Positive Rate (TPR)": (
        "The true positive rate is the number of True Positives that are "
        "predicted as positive as a proportion of the total number of positives. "
        "If an attacker has N examples that were actually in the training set, "
        "the TPR is the proportion of these that they predict as being in the "
        "training set."
    ),
    "ACC": "The proportion of predictions that the attacker makes that are correct.",
}


def write_json(output: dict, dest: str) -> None:
    """Write attack report to JSON."""
    attack_formatter = GenerateJSONModule(dest + ".json")
    attack_report: str = json.dumps(output, cls=CustomJSONEncoder)
    attack_name: str = output["metadata"]["attack_name"]
    attack_formatter.add_attack_output(attack_report, attack_name)


class CustomJSONEncoder(json.JSONEncoder):
    """JSON encoder that can cope with numpy arrays, etc."""

    def default(self, o: Any):
        """If an object is an np.ndarray, convert to list."""
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        try:  # Try the default method first
            return super().default(o)
        except TypeError:
            return str(o)  # If object is not serializable, convert it to a string


def _write_dict(pdf: FPDF, data: dict, border: int = BORDER) -> None:
    """Write a dictionary to the pdf."""
    for key, value in data.items():
        pdf.set_font("arial", "B", 14)
        pdf.cell(0, 5, key, border, 1, "L")
        pdf.set_font("arial", "", 12)
        pdf.multi_cell(0, 5, str(value), 0, 1)
        pdf.ln(h=5)


def title(
    pdf: FPDF,
    text: str,
    border: int = BORDER,
    font_size: int = 24,
    font_style: str = "B",
) -> None:
    """Write a title block."""
    pdf.set_font("arial", font_style, font_size)
    pdf.ln(h=5)
    pdf.cell(0, 0, text, border, 1, "C")
    pdf.ln(h=5)


def subtitle(  # pylint: disable = too-many-arguments
    pdf: FPDF,
    text: str,
    indent: int = 10,
    border: int = BORDER,
    font_size: int = 12,
    font_style: str = "B",
) -> None:
    """Write a subtitle block."""
    pdf.cell(indent, border=border)
    pdf.set_font("arial", font_style, font_size)
    pdf.cell(75, 10, text, border, 1)


def line(  # pylint: disable = too-many-arguments
    pdf: FPDF,
    text: str,
    indent: int = 0,
    border: int = BORDER,
    font_size: int = 11,
    font_style: str = "",
    font: str = "arial",
) -> None:
    """Write a standard block."""
    if indent > 0:
        pdf.cell(indent, border=border)
    pdf.set_font(font, font_style, font_size)
    pdf.multi_cell(0, 5, text, border, 1)


def _roc_plot_single(metrics: dict, save_name: str) -> None:
    """Create a roc_plot for a single experiment."""
    plt.figure()
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(metrics["fpr"], metrics["tpr"], "r", linewidth=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.tight_layout()
    plt.savefig(save_name)


def _roc_plot(metrics: dict, save_name: str) -> None:
    """Create a roc plot for multiple repetitions."""
    plt.figure()
    plt.plot([0, 1], [0, 1], "k--")

    # Compute average ROC
    base_fpr = np.linspace(0, 1, 1000)
    all_tpr = np.zeros((len(metrics), len(base_fpr)), float)
    for i, metric_set in enumerate(metrics):
        all_tpr[i, :] = np.interp(base_fpr, metric_set["fpr"], metric_set["tpr"])

    for _, metric_set in enumerate(metrics):
        plt.plot(
            metric_set["fpr"], metric_set["tpr"], color="lightsalmon", linewidth=0.5
        )

    tpr_mu = all_tpr.mean(axis=0)
    plt.plot(base_fpr, tpr_mu, "r")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.grid()
    plt.savefig(save_name)


def create_mia_report(attack_output: dict) -> FPDF:
    """Make a worst case membership inference report.

    Parameters
    ----------
    attack_output : dict
        Dictionary with the following items:

        metadata : dict
            Dictionary of metadata.
        attack_experiment_logger : dict
            List of metrics as dictionary items for an experiment.
        dummy_attack_experiment_logger : dict
            List of metrics as dictionary items across dummy experiments.

    Returns
    -------
    pdf : fpdf.FPDF
        fpdf document object
    """
    mia_metrics = [
        v
        for _, v in attack_output["attack_experiment_logger"][
            "attack_instance_logger"
        ].items()
    ]
    metadata = attack_output["metadata"]

    path: str = metadata["attack_params"]["output_dir"]
    dest_log_roc = os.path.join(path, "log_roc.png")
    _roc_plot(mia_metrics, dest_log_roc)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_xy(0, 0)
    title(pdf, "WorstCase MIA attack result report")
    subtitle(pdf, "Introduction")
    line(pdf, INTRODUCTION)
    subtitle(pdf, "Experiment summary")
    for key, value in metadata["attack_params"].items():
        line(pdf, f"{key:>30s}: {str(value):30s}", font="courier")
    subtitle(pdf, "Global metrics")
    for key, value in metadata["global_metrics"].items():
        line(pdf, f"{key:>30s}: {str(value):30s}", font="courier")

    subtitle(pdf, "Metrics")
    line(
        pdf,
        "The following show summaries of the attack metrics over the repetitions",
        font="arial",
    )
    for metric in DISPLAY_METRICS:
        vals = np.array([m[metric] for m in mia_metrics])
        if metric in MAPPINGS:
            vals = np.array([MAPPINGS[metric](v) for v in vals])
        text = (
            f"{metric:>12} mean = {vals.mean():.2f}, var = {vals.var():.4f}, "
            f"min = {vals.min():.2f}, max = {vals.max():.2f}"
        )
        line(pdf, text, font="courier")

    _add_log_roc_to_page(dest_log_roc, pdf)
    line(pdf, LOGROC_CAPTION)

    pdf.add_page()
    title(pdf, "Glossary")
    _write_dict(pdf, GLOSSARY)

    if os.path.exists(dest_log_roc):
        os.remove(dest_log_roc)
    return pdf


def write_pdf(report_dest: str, pdf_report: FPDF) -> None:
    """Create pdf and append contents if it already exists."""
    if os.path.exists(report_dest + ".pdf"):
        old_pdf = report_dest + ".pdf"
        new_pdf = report_dest + "_new.pdf"
        pdf_report.output(new_pdf)
        merger = PdfWriter()
        for pdf in [old_pdf, new_pdf]:
            merger.append(pdf)
        merger.write(old_pdf)
        merger.close()
        os.remove(new_pdf)
    else:
        pdf_report.output(report_dest + ".pdf")


def _add_log_roc_to_page(log_roc: str = None, pdf_obj: FPDF = None) -> None:
    if log_roc is not None:
        pdf_obj.add_page()
        subtitle(pdf_obj, "Log ROC")
        pdf_obj.image(log_roc, x=None, y=None, w=0, h=140, type="", link="")
        pdf_obj.set_font("arial", "", 12)


def _plot_lira_individuals(metrics: dict, dest: str) -> None:
    """Create a plot of the individual record LiRA scores."""
    scores = np.array(metrics["individual"]["score"])
    member = np.array(metrics["individual"]["member"])

    _, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # members
    mask = member == 1
    y_train = scores[mask]
    x_train = np.arange(y_train.shape[0])

    sorted_indicies = np.argsort(y_train)
    y_sorted = y_train[sorted_indicies]

    axes[0].scatter(x_train, y_sorted, color="b", s=2, label="LiRA Probability")
    axes[0].set_title("Member Records")
    axes[0].set_xlabel("Record (sorted)")
    axes[0].legend(loc=0)

    # nonmembers
    y_test = scores[~mask]
    x_test = np.arange(y_test.shape[0])

    sorted_indicies = np.argsort(y_test)
    y_sorted = y_test[sorted_indicies]

    axes[1].scatter(x_test, y_sorted, color="r", s=2, label="LiRA Probability")
    axes[1].set_title("Nonmember Records")
    axes[1].set_xlabel("Record (sorted)")
    axes[1].legend(loc=0)

    plt.tight_layout()
    plt.savefig(dest)


def create_lr_report(output: dict) -> FPDF:
    """Make a lira membership inference report.

    Parameters
    ----------
    output : dict
        Dictionary with the following items:

        metadata : dict
            Dictionary of metadata.

        attack_experiment_logger : dict
            List of metrics as dictionary items for an experiments.
            In case of LiRA attack scenario, this will have dictionary items of
            `attack_instance_logger` that will have a single metrics dictionary.

    Returns
    -------
    pdf : fpdf.FPDF
        fpdf document object
    """
    mia_metrics = [
        v
        for _, v in output["attack_experiment_logger"]["attack_instance_logger"].items()
    ][0]
    metadata = output["metadata"]

    path: str = metadata["attack_params"]["output_dir"]
    dest_log_roc = os.path.join(path, "log_roc.png")
    _roc_plot_single(mia_metrics, dest_log_roc)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_xy(0, 0)
    title(pdf, "Likelihood Ratio Attack Report")
    subtitle(pdf, "Introduction")
    subtitle(pdf, "Metadata")
    for key, value in metadata["attack_params"].items():
        line(pdf, f"{key:>30s}: {str(value):30s}", font="courier")
    for key, value in metadata["global_metrics"].items():
        line(pdf, f"{key:>30s}: {str(value):30s}", font="courier")
    subtitle(pdf, "Metrics")
    sub_metrics_dict = {
        key: val for key, val in mia_metrics.items() if isinstance(val, float)
    }
    for key, value in sub_metrics_dict.items():
        val = MAPPINGS[key](value) if key in MAPPINGS else value
        line(pdf, f"{key:>30s}: {val:.4f}", font="courier")

    pdf.add_page()
    subtitle(pdf, "ROC Curve")
    pdf.image(dest_log_roc, x=None, y=None, w=0, h=140, type="", link="")

    dest_ind_plot = os.path.join(path, "lira_individual.png")
    if "individual" in mia_metrics:
        _plot_lira_individuals(mia_metrics, dest_ind_plot)
        pdf.add_page()
        subtitle(pdf, "Individual LiRA Scores")
        pdf.image(dest_ind_plot, x=None, y=None, w=180, h=90, type="", link="")

    # clean up
    files = [dest_log_roc, dest_ind_plot]
    for file in files:
        if os.path.exists(file):
            os.remove(file)
    return pdf
