
"""
Base classes and shared functionality for report visualisations.

This module defines the abstract ``Visualisation`` class, which provides
a common interface for all visualisations used in the reporting pipeline.

Visualisations are responsible for:
- determining whether they apply to a given experiment
- generating plots based on experiment data
- saving those plots to disk

Concrete visualisations should subclass ``Visualisation`` and implement:
- ``visualisation_applies_to_experiment()``
- ``plot()``

This module also includes shared helper methods for common plotting tasks.
"""

from __future__ import annotations

# Standard library
import os
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# Third-party
import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from sacroml.reporting.models import ReportExperiment


class Visualisation(ABC):
    """
    Base class for experiment visualisations.

    A visualisation is associated with a single experiment and is responsible
    for generating plots derived from experiment data.
    """

    #: Short tag used in filenames and identifiers (must be overridden)
    visualisation_tag: str

    #: Human-readable title for the visualisation
    title: str | None = None

    #: Caption text describing the visualisation
    caption: str | None = None

    def __init__(self, experiment: ReportExperiment) -> None:
        """
        Initialise the visualisation.

        Parameters
        ----------
        experiment : ReportExperiment
            Experiment this visualisation is based on.
        """
        self.experiment: ReportExperiment = experiment

    def plot_name(self) -> str:
        """
        Generate a safe filename for the plot.

        Returns
        -------
        str
            Filename including extension.
        """
        safe_experiment_id = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            self.experiment.id,
        )
        return f"{self.visualisation_tag}_{safe_experiment_id}.png"

    def plot_path(self) -> str:
        """
        Return the relative path for the plot image.

        Returns
        -------
        str
            Relative path under the figures directory.
        """
        return os.path.join("figures", self.plot_name())

    @abstractmethod
    def visualisation_applies_to_experiment(self) -> bool:
        """
        Determine whether this visualisation applies to the experiment.

        Subclasses must implement this method.

        Returns
        -------
        bool
            True if the visualisation can be generated.
        """
        ...

    @abstractmethod
    def plot(self, output_dir: str) -> str:
        """
        Generate and save the visualisation plot.

        Subclasses must implement this method.

        Parameters
        ----------
        output_dir : str
            Directory where the plot should be written.

        Returns
        -------
        str
            Markdown-relative path to the generated image.
        """
        ...

    def _plot_instance_curves_with_mean(
        self,
        x_key: str,
        y_key: str,
        base_x: np.ndarray,
        *,
        instance_color: str = "lightsalmon",
        mean_color: str = "red",
        instance_lw: float = 0.5,
        mean_lw: float = 2.0,
    ) -> None:
        """
        Plot per-instance curves and their mean, interpolated onto a shared x-grid.
        """
        y_curves = self.experiment.instance_results(y_key)
        x_curves = self.experiment.instance_results(x_key)

        assert len(x_curves) == len(y_curves)

        all_y = np.zeros((len(x_curves), base_x.size), dtype=float)

        for i in range(len(x_curves)):
            all_y[i, :] = np.interp(base_x, x_curves[i], y_curves[i])

        for i in range(len(x_curves)):
            plt.plot(
                x_curves[i],
                y_curves[i],
                color=instance_color,
                linewidth=instance_lw,
            )

        mean_y = all_y.mean(axis=0)
        plt.plot(base_x, mean_y, color=mean_color, linewidth=mean_lw)
