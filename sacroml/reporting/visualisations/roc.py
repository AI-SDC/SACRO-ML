
"""
ROC visualisation for reporting.

This module provides the ``ROCPlot`` visualisation, which generates
Receiver Operating Characteristic (ROC) plots for experiments that
include false positive rate (FPR) and true positive rate (TPR) metrics.

The plot shows per-instance ROC curves along with their mean, displayed
on log-scaled axes to emphasise regions of low false positive rate.
"""

from __future__ import annotations


import os

import numpy as np
import matplotlib.pyplot as plt

from sacroml.reporting.visualisations.base import Visualisation


class ROCPlot(Visualisation):
    """
    Log-scale ROC (Receiver Operating Characteristic) plot.
    """

    visualisation_tag: str = "roc"
    title: str = "Log ROC"

    caption: str = (
        "This plot shows the False Positive Rate (x) versus the True Positive Rate "
        "(y). The axes are in log space enabling focus on regions where the false "
        "positive rate is low. Curves consistently above the y = x line indicate "
        "potential disclosure risk."
    )

    def visualisation_applies_to_experiment(self) -> bool:
        """
        Check whether the experiment contains the required metrics.

        Returns
        -------
        bool
            True if both FPR and TPR metrics are present.
        """
        return {"fpr", "tpr"} <= set(self.experiment.instance_metric_keys())

    def plot(self, output_dir: str) -> str:
        """
        Generate and save the ROC plot.

        Parameters
        ----------
        output_dir : str
            Directory where the plot image should be saved.

        Returns
        -------
        str
            Markdown-relative path to the saved image.
        """
        os.makedirs(output_dir, exist_ok=True)

        filename = self.plot_name()
        fs_path = os.path.join(output_dir, filename)

        plt.figure()
        plt.plot([0, 1], [0, 1], "k--")

        base_fpr = np.linspace(0.0, 1.0, 1000)
        self._plot_instance_curves_with_mean(
            x_key="fpr",
            y_key="tpr",
            base_x=base_fpr,
        )

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid()
        plt.tight_layout()
        plt.savefig(fs_path)
        plt.close()

        return f"figures/{filename}"
    