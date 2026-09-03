"""
Attribute inference visualisations.

This module provides visualisations for attribute inference attack results,
including:

- Quantitative risk plots
- Categorical risk (improvement over baseline)
- Categorical coverage (fraction of dataset inferred)

These visualisations operate on experiment instance results and produce
bar charts to help interpret attack effectiveness and data exposure.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

from sacroml.reporting.visualisations.base import Visualisation



class QuantitativeRiskPlot(Visualisation):
    """Bar chart of quantitative attribute risk scores."""

    visualisation_tag = "quant_risk"
    title = "Quantitative Risk"

    def visualisation_applies_to_experiment(self) -> bool:
        return "quantitative" in self.experiment.instance_metric_keys()

    def plot(self, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)

        instance = next(iter(self.experiment.instances.values()))
        results = instance.results["quantitative"].value

        if not results:
            return ""

        x = np.arange(len(results))
        names = [f["name"] for f in results]
        train = [f["train"] * 100 for f in results]
        test = [f["test"] * 100 for f in results]

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(x + 0.2, train, 0.4, label="train")
        ax.bar(x - 0.2, test, 0.4, label="test")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=90)
        ax.set_ylim([0, 100])
        ax.set_title(self.title)
        ax.grid(linestyle="dotted")
        ax.legend()

        plt.tight_layout()

        path = os.path.join(output_dir, self.plot_name())
        plt.savefig(path)
        plt.close()

        return f"figures/{self.plot_name()}"


class CategoricalRiskPlot(Visualisation):
    """
    Bar chart showing improvement over baseline for categorical attributes.
    """

    visualisation_tag = "cat_risk"
    title = "Categorical Risk (Improvement over baseline)"

    def visualisation_applies_to_experiment(self) -> bool:
        return "categorical" in self.experiment.instance_metric_keys()

    def plot(self, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)

        instance = next(iter(self.experiment.instances.values()))
        results = instance.results["categorical"].value

        if not results:
            return ""

        x = np.arange(len(results))
        names = []
        train_vals = []
        test_vals = []

        for feature in results:
            names.append(feature["name"])

            c_a, t_a, b_a, _, _ = feature["train"]
            c_b, t_b, b_b, _, _ = feature["test"]

            train_val = ((c_a / t_a) * 100 - b_a) if t_a > 0 else 0
            test_val = ((c_b / t_b) * 100 - b_b) if t_b > 0 else 0

            train_vals.append(train_val)
            test_vals.append(test_val)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(x + 0.2, train_vals, 0.4, label="train")
        ax.bar(x - 0.2, test_vals, 0.4, label="test")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=90)
        ax.set_ylim([-100, 100])
        ax.set_title(self.title)
        ax.grid(linestyle="dotted")
        ax.legend()

        plt.tight_layout()

        path = os.path.join(output_dir, self.plot_name())
        plt.savefig(path)
        plt.close()

        return f"figures/{self.plot_name()}"


class CategoricalFractionPlot(Visualisation):
    """Bar chart showing fraction of dataset inferred for categorical attributes."""

    visualisation_tag = "cat_frac"
    title = "Categorical Coverage (%)"

    def visualisation_applies_to_experiment(self) -> bool:
        return "categorical" in self.experiment.instance_metric_keys()

    def plot(self, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)

        instance = next(iter(self.experiment.instances.values()))
        results = instance.results["categorical"].value

        if not results:
            return ""

        x = np.arange(len(results))
        names = []
        train_vals = []
        test_vals = []

        for feature in results:
            names.append(feature["name"])

            _, total_a, _, _, n_a = feature["train"]
            _, total_b, _, _, n_b = feature["test"]

            train_val = (total_a / n_a) * 100 if n_a > 0 else 0
            test_val = (total_b / n_b) * 100 if n_b > 0 else 0

            train_vals.append(train_val)
            test_vals.append(test_val)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(x + 0.2, train_vals, 0.4, label="train")
        ax.bar(x - 0.2, test_vals, 0.4, label="test")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=90)
        ax.set_ylim([0, 100])
        ax.set_title(self.title)
        ax.grid(linestyle="dotted")
        ax.legend()

        plt.tight_layout()

        path = os.path.join(output_dir, self.plot_name())
        plt.savefig(path)
        plt.close()

        return f"figures/{self.plot_name()}"