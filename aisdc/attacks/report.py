"""Code for automatic report generation"""
import json

import numpy as np
import pylab as plt
from fpdf import FPDF

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
    "This report provides a summary of a series of simulated attack experiments performed "
    "on the model outputs provided. An attack model is trained to attempt to distinguish "
    "between outputs from training (in-sample) and testing (out-of-sample) data. The metrics "
    "below describe the success of this classifier. A successful classifier indicates that the "
    "original model is unsafe and should not be allowed to be released from the TRE.\n"
    "In particular, the simulation splits the data provided into test and train sets (each will "
    "in- and out-of-sample examples). The classifier is trained on the train set and evaluated "
    "on the test set. This is repeated with different train/test splits a user-specified number "
    "of times.\n"
    "To help place the results in context, the code may also have run a series of baseline "
    "experiments. In these, random model outputs for hypothetical in- and out-of-sample data are "
    "generated with identical statistical properties. In these baseline cases, there is no signal "
    "that an attacker could leverage and therefore these values provide a baseline against "
    "which the actual values can be compared.\n"
    "For some metrics (FDIF and AUC), we are able to compute p-values. In each case, shown below "
    "(in the Global metrics sections) is the number of repetitions that exceeded the p-value "
    "threshold both without, and with correction for multiple testing (Benjamini-Hochberg "
    "procedure).\n"
    "ROC curves for all real (red) and dummy (blue) repetitions are provided. These are shown in "
    "log space (as reommended here [ADD URL]) to emphasise the region in which risk is highest -- "
    "the bottom left (are high true positive rates possible with low false positive rates).\n"
    "A description of the metrics and how to interpret them within the context of an attack is "
    "given below."
)

LOGROC_CAPTION = (
    "This plot shows the False Positive Rate (x) versus the True Positive Rate (y). "
    "The axes are in log space enabling us to focus on areas where the False Positive Rate is low "
    "(left hand area). Curves above the y = x line (black dashes) in this region represent a "
    "disclosure risk as an attacker can obtain many more true than false positives. "
    "The solid coloured lines show the curves for the attack simulations with the true model "
    "outputs. The lighter grey lines show the curves for randomly generated outputs with "
    "no structure (i.e. in- and out-of- sample predictions are generated from the same "
    "distributions. Solid curves consistently higher than the grey curves in the left hand "
    "part of the plot are a sign of concern."
)

GLOSSARY = {
    "AUC": "Area Under the ROC curve",
    "True Positive Rate (TPR)": (
        "The true positive rate is the number of True Positives that are predicted as positive as "
        "a proportion of the total number of positives. If an attacker has N examples that were "
        "actually in the training set, the TPR is the proportion of these that they predict as "
        "being in the training set."
    ),
    "ACC": "The proportion of predictions that the attacker makes that are correct.",
}


class NumpyArrayEncoder(json.JSONEncoder):
    """Json encoder that can cope with numpy arrays"""

    def default(self, o):
        """If an object is an np.ndarray, convert to list"""
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.int64):
            return int(o)
        if isinstance(o, np.int32):
            return int(o)
        return json.JSONEncoder.default(self, o)


def _write_dict(pdf, input_dict, indent=0, border=BORDER):
    """Write a dictionary to the pdf"""
    for key, value in input_dict.items():
        pdf.set_font("arial", "B", 14)
        pdf.cell(75, 5, key, border, 1, "L")
        pdf.cell(indent, 0)
        pdf.set_font("arial", "", 12)
        pdf.multi_cell(150, 5, value, border, "L")
        pdf.ln(h=5)


def title(pdf, text, border=BORDER, font_size=24, font_style="B"):
    """Write a title block"""
    pdf.set_font("arial", font_style, font_size)
    pdf.ln(h=5)
    pdf.cell(0, 0, text, border, 1, "C")
    pdf.ln(h=5)


def subtitle(
    pdf, text, indent=10, border=BORDER, font_size=12, font_style="B"
):  # pylint: disable = too-many-arguments
    """Write a subtitle block"""
    pdf.cell(indent, border=border)
    pdf.set_font("arial", font_style, font_size)
    pdf.cell(75, 10, text, border, 1)


def line(
    pdf, text, indent=0, border=BORDER, font_size=11, font_style="", font="arial"
):  # pylint: disable = too-many-arguments
    """Write a standard block"""
    if indent > 0:
        pdf.cell(indent, border=border)
    pdf.set_font(font, font_style, font_size)
    pdf.multi_cell(0, 5, text, border, 1)


def _roc_plot_single(metrics, save_name):
    """Create a roc_plot for a single experiment"""
    plt.figure()
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(metrics["fpr"], metrics["tpr"], "r", linewidth=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.tight_layout()
    plt.savefig(save_name)


def _roc_plot(metrics, dummy_metrics, save_name):
    """Create a roc plot for multiple repetitions"""
    plt.figure()
    plt.plot([0, 1], [0, 1], "k--")
    if dummy_metrics is None or len(dummy_metrics) == 0:
        do_dummy = False
    else:
        do_dummy = True

    # Compute average ROC
    base_fpr = np.linspace(0, 1, 1000)
    # base_fpr = np.logspace(-4, 0, 1000)
    all_tpr = np.zeros((len(metrics), len(base_fpr)), float)
    for i, metric_set in enumerate(metrics):
        all_tpr[i, :] = np.interp(base_fpr, metric_set["fpr"], metric_set["tpr"])

    if do_dummy:
        all_tpr_dummy = np.zeros((len(dummy_metrics), len(base_fpr)), float)
        for i, metric_set in enumerate(dummy_metrics):
            all_tpr_dummy[i, :] = np.interp(
                base_fpr, metric_set["fpr"], metric_set["tpr"]
            )

        for _, metric_set in enumerate(dummy_metrics):
            plt.plot(
                metric_set["fpr"],
                metric_set["tpr"],
                color="lightsteelblue",
                linewidth=0.5,
                alpha=0.5,
            )

    for _, metric_set in enumerate(metrics):
        plt.plot(
            metric_set["fpr"], metric_set["tpr"], color="lightsalmon", linewidth=0.5
        )

    tpr_mu = all_tpr.mean(axis=0)
    plt.plot(base_fpr, tpr_mu, "r")

    if do_dummy:
        dummy_mu = all_tpr_dummy.mean(axis=0)
        plt.plot(base_fpr, dummy_mu, "b")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.grid()
    plt.savefig(save_name)


def create_mia_report(attack_output: dict) -> FPDF:
    """make a worst case membership inference report

    Parameters
    ----------

    metadata: dict
        dictionary of metadata

    mia_metrics: list
        list of metrics dictionaries

    dummy_metrics: list
        list of metrics dictionaries

    dummy_metadata: dict
        metadata for dummy experiments

    Returns
    -------

    pdf: fpdf.FPDF
        fpdf document object

    """

    dummy_metrics = attack_output["dummy_attack_metrics"]
    mia_metrics = attack_output["attack_metrics"]
    metadata = attack_output["metadata"]
    if dummy_metrics is None or len(dummy_metrics) == 0:
        do_dummy = False
    else:
        do_dummy = True

    _roc_plot(mia_metrics, dummy_metrics, "log_roc.png")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_xy(0, 0)
    title(pdf, "WorstCase MIA attack result report")
    subtitle(pdf, "Introduction")
    line(pdf, INTRODUCTION)
    subtitle(pdf, "Experiment summary")
    for key, value in metadata["experiment_details"].items():
        line(pdf, f"{key:>30s}: {str(value):30s}", font="courier")
    subtitle(pdf, "Global metrics")
    for key, value in metadata["global_metrics"].items():
        line(pdf, f"{key:>30s}: {str(value):30s}", font="courier")
    if do_dummy:
        subtitle(pdf, "Baseline global metrics")
        for key, value in metadata["baseline_global_metrics"].items():
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

    if do_dummy:
        subtitle(pdf, "Baseline metrics")
        line(
            pdf,
            (
                "The following show summaries of the attack metrics over the "
                "repetitions where there is no statistical difference between "
                "predictions in the training and test sets. Simulation was done "
                "with training and test set sizes equal to the real ones"
            ),
            font="arial",
        )
        for metric in DISPLAY_METRICS:
            vals = np.array([m[metric] for m in dummy_metrics])
            if metric in MAPPINGS:
                vals = np.array([MAPPINGS[metric](v) for v in vals])
            text = (
                f"{metric:>12} mean = {vals.mean():.2f}, var = {vals.var():.4f}, "
                f"min = {vals.min():.2f}, max = {vals.max():.2f}"
            )
            line(pdf, text, font="courier")

    pdf.add_page()
    subtitle(pdf, "Log ROC")
    pdf.image("log_roc.png", x=None, y=None, w=0, h=140, type="", link="")
    pdf.set_font("arial", "", 12)
    line(pdf, LOGROC_CAPTION)

    pdf.add_page()
    title(pdf, "Glossary")
    _write_dict(pdf, GLOSSARY)

    return pdf


def create_json_report(output):
    """Create a report in json format for injestion by other tools"""
    # Initial work, just dump mia_metrics and dummy_metrics into a json structure
    return json.dumps(output, cls=NumpyArrayEncoder)


def create_lr_report(output):
    """TODO"""
    mia_metrics = output["attack_metrics"][0]
    metadata = output["metadata"]
    _roc_plot_single(mia_metrics, "log_roc.png")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_xy(0, 0)
    title(pdf, "Likelihood Ratio Attack Report")
    subtitle(pdf, "Introduction")
    subtitle(pdf, "Metadata")
    for key, value in metadata["experiment_details"].items():
        line(pdf, f"{key:>30s}: {str(value):30s}", font="courier")
    for key, value in metadata["global_metrics"].items():
        line(pdf, f"{key:>30s}: {str(value):30s}", font="courier")
    subtitle(pdf, "Metrics")
    sub_metrics_dict = {
        key: val for key, val in mia_metrics.items() if isinstance(val, float)
    }
    for key, value in sub_metrics_dict.items():
        if key in MAPPINGS:
            value = MAPPINGS[key](value)
        line(pdf, f"{key:>30s}: {value:.4f}", font="courier")
    pdf.add_page()
    subtitle(pdf, "ROC Curve")
    pdf.image("log_roc.png", x=None, y=None, w=0, h=140, type="", link="")
    return pdf
