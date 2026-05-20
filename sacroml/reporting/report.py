"""Code for automatic report generation."""

from __future__ import annotations

import datetime
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Type, Union
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import yaml
from jinja2 import Environment, FileSystemLoader
from importlib import resources

class ReportMetric:
    """
    Represents a metric definition used in a report.

    A metric describes how a quantity is measured, including its label,
    description, units, and how aggregated statistics may be computed.
    """

    def __init__(
        self,
        name: str,
        label: str,
        description: str,
        units: Optional[str],
        higher_is_better: bool,
        category: Optional[str],
        typical_range: Optional[str],
        notes: Optional[str],
        allowed_aggregations: Sequence[str],
    ) -> None:
        """
        Initialise a report metric.

        Parameters
        ----------
        name : str
            Internal identifier for the metric.

        label : str
            Human-readable name of the metric.

        description : str
            Detailed description explaining what the metric measures.

        units : str or None
            Units of measurement (e.g. "seconds", "accuracy"). None if not applicable.

        higher_is_better : bool
            Indicates whether a higher value represents better performance.

        category : str or None
            Optional category grouping for the metric (e.g. "performance", "privacy").

        typical_range : str or None
            Expected or typical range of values for the metric.

        notes : str or None
            Additional notes or contextual information about the metric.

        allowed_aggregations : Sequence[str]
            List of aggregation types allowed for this metric
            (e.g. ["mean", "min", "max", "var"]).
        """
        self.name: str = name
        self.label: str = label
        self.description: str = description
        self.units: Optional[str] = units
        self.higher_is_better: bool = higher_is_better
        self.category: Optional[str] = category
        self.typical_range: Optional[str] = typical_range
        self.notes: Optional[str] = notes
        self.allowed_aggregations: list[str] = list(allowed_aggregations)
    
    def __str__(self) -> str:
        """
        Return a user-friendly string representation.

        Returns
        -------
        str
            Human-readable representation of the metric.
        """
        return f"ReportMetric: {self.label}"

    def __repr__(self) -> str:
        """
        Return a detailed string representation for debugging.

        Returns
        -------
        str
            Unambiguous representation including key attributes.
        """
        return (
            f"ReportMetric("
            f"name='{self.name}', "
            f"label='{self.label}', "
            f"description='{self.description}', "
            f"units='{self.units}', "
            f"higher_is_better={self.higher_is_better}, "
            f"category='{self.category}', "
            f"typical_range='{self.typical_range}'"
            f")"
        )





class ReportResult:
    """
    Represents the value of a metric for a specific context.

    A result associates a `ReportMetric` definition with an observed value.
    The value may take one of several forms:

    - A scalar (e.g. float, int, string) for global metrics
    - A dictionary of aggregated statistics (e.g. mean, min, max, var)
    - A list of numeric values (e.g. per-instance values or curve data)
    """

    def __init__(
        self,
        metric: ReportMetric,
        value: Union[Any, Dict[str, float], List[float]],
    ) -> None:
        """
        Initialise a report result.

        Parameters
        ----------
        metric : ReportMetric
            The metric definition associated with this result.

        value : Any or dict[str, float] or list[float]
            The value of the metric.

            - Scalar → global metric
            - dict → aggregated statistics:
              {"mean": ..., "min": ..., "max": ..., "var": ...}
            - list[float] → sequence of numeric values (e.g. per-instance
              measurements or curve data such as ROC points)
        """
        self.metric: ReportMetric = metric
        self.value: Union[Any, Dict[str, float], List[float]] = value

    def is_aggregate(self) -> bool:
        """
        Determine whether this result represents aggregated statistics.

        Returns
        -------
        bool
            True if the value is a dictionary of aggregations.
        """
        return isinstance(self.value, dict)

    def is_sequence(self) -> bool:
        """
        Determine whether this result represents a sequence of numeric values.

        Returns
        -------
        bool
            True if the value is a list of floats.
        """
        return isinstance(self.value, list)

    def is_scalar(self) -> bool:
        """
        Determine whether this result is a scalar value.

        Returns
        -------
        bool
            True if the value is neither a dict nor a list.
        """
        return not isinstance(self.value, (dict, list))

    def __repr__(self) -> str:
        """
        Return a detailed string representation for debugging.

        Returns
        -------
        str
            Unambiguous representation of the result.
        """
        return (
            f"ReportResult("
            f"metric='{self.metric}', "
            f"value='{self.value}'"
            f")"
        )

class ReportParameter:
    """
    Represents a parameter definition for an experiment or attack.

    A parameter defines a configurable input to an experiment, including
    its internal name, display label, and description.
    """

    def __init__(self, name: str, label: str, description: str) -> None:
        """
        Initialise a report parameter.

        Parameters
        ----------
        name : str
            Internal identifier for the parameter.

        label : str
            Human-readable name for display in reports.

        description : str
            Description explaining the purpose and meaning of the parameter.
        """
        self.name: str = name
        self.label: str = label
        self.description: str = description

    def __str__(self) -> str:
        """
        Return a user-friendly string representation.

        Returns
        -------
        str
            Human-readable representation of the parameter.
        """
        return f"ReportParameter: {self.label}"

    def __repr__(self) -> str:
        """
        Return a detailed string representation for debugging.

        Returns
        -------
        str
            Unambiguous representation including all fields.
        """
        return (
            f"ReportParameter("
            f"name='{self.name}', "
            f"label='{self.label}', "
            f"description='{self.description}'"
            f")"
        )
    

class ReportParameterInstance:
    """
    Represents a specific value assigned to a parameter for an experiment.

    This class links a `ReportParameter` definition with a concrete value
    used in a particular experiment instance.
    """

    def __init__(self, parameter: ReportParameter, value: Any) -> None:
        """
        Initialise a parameter instance.

        Parameters
        ----------
        parameter : ReportParameter
            The parameter definition.

        value : Any
            The value assigned to the parameter. This may be of any type
            (e.g. int, float, str, bool) depending on the parameter.
        """
        self.parameter: ReportParameter = parameter
        self.value: Any = value

    def __repr__(self) -> str:
        """
        Return a detailed string representation for debugging.

        Returns
        -------
        str
            Unambiguous representation including parameter and value.
        """
        return (
            f"ReportParameterInstance("
            f"parameter='{self.parameter}', "
            f"value='{self.value}'"
            f")"
        )


class ReportInstance:
    """
    Represents a single instance within an experiment.

    Each instance contains a set of metric results corresponding to a
    specific run, sample, or evaluation unit within an experiment.
    """

    def __init__(
        self,
        id: str,
        number: int,
        results: Dict[str, ReportResult],
    ) -> None:
        """
        Initialise a report instance.

        Parameters
        ----------
        id : str
            Unique identifier for the instance.

        number : int
            Numerical index of the instance (e.g. extracted from ID).

        results : dict[str, ReportResult]
            Mapping of metric names to their corresponding results
            for this instance.
        """
        self.id: str = id
        self.number: int = number
        self.results: Dict[str, ReportResult] = results

    def __repr__(self) -> str:
        """
        Return a detailed string representation for debugging.

        Returns
        -------
        str
            Unambiguous representation including ID and number.
        """
        return (
            f"ReportInstance("
            f"id='{self.id}', "
            f"number={self.number}, "
            f"results={self.results}"
            f")"
        )

class ReportAttackCategory:
    """
    Represents a high-level category grouping related attack types.

    Attack categories are used to organise attacks in the report and
    control their ordering in the final output.
    """

    def __init__(
        self,
        name: str,
        label: str,
        description: Optional[str],
        order: int,
    ) -> None:
        """
        Initialise an attack category.

        Parameters
        ----------
        name : str
            Internal identifier for the attack category.

        label : str
            Human-readable name for display in the report.

        description : str or None
            Optional description of the attack category.

        order : int
            Sort order used when rendering categories in the report.
            Lower values are rendered first.
        """
        self.name: str = name
        self.label: str = label
        self.description: Optional[str] = description
        self.order: int = order

    def __repr__(self) -> str:
        """
        Return a detailed string representation for debugging.

        Returns
        -------
        str
            Unambiguous representation including all fields.
        """
        return (
            f"ReportAttackCategory("
            f"name='{self.name}', "
            f"label='{self.label}', "
            f"description='{self.description}', "
            f"order={self.order}"
            f")"
        )

class ReportAttackType:
    """
    Represents a specific type of attack within an attack category.

    An attack type groups together experiments that share the same attack
    definition, parameters, and key metrics.
    """

    def __init__(
        self,
        name: str,
        label: str,
        description: Optional[str],
        category: Optional[ReportAttackCategory],
        parameters: Dict[str, ReportParameter],
        key_metrics: Dict[str, ReportMetric],
    ) -> None:
        """
        Initialise an attack type.

        Parameters
        ----------
        name : str
            Internal identifier for the attack type.

        label : str
            Human-readable name for display in the report.

        description : str or None
            Optional description explaining the attack type.

        category : ReportAttackCategory or None
            The category this attack type belongs to.

        parameters : dict[str, ReportParameter]
            Mapping of parameter names to parameter definitions relevant
            for this attack type.

        key_metrics : dict[str, ReportMetric]
            Mapping of metric names to metric definitions that should be
            highlighted or aggregated for this attack type.
        """
        self.name: str = name
        self.label: str = label
        self.description: Optional[str] = description
        self.category: Optional[ReportAttackCategory] = category
        self.parameters: Dict[str, ReportParameter] = parameters
        self.key_metrics: Dict[str, ReportMetric] = key_metrics

    def __str__(self) -> str:
        """
        Return a user-friendly string representation.

        Returns
        -------
        str
            Human-readable representation of the attack type.
        """
        return f"ReportAttackType: {self.label}"

    def __repr__(self) -> str:
        """
        Return a detailed string representation for debugging.

        Returns
        -------
        str
            Unambiguous representation including all fields.
        """
        return (
            f"ReportAttackType("
            f"name='{self.name}', "
            f"label='{self.label}', "
            f"description='{self.description}', "
            f"category='{self.category}', "
            f"parameters={list(self.parameters.keys())}, "
            f"key_metrics={list(self.key_metrics.keys())}"
            f")"
        )


class ReportExperiment:
    """
    Represents a single experiment within the report.

    An experiment ties together:
    - an attack type
    - parameter values
    - per-instance metric results
    - aggregated and global metrics
    - generated visualisations
    """

    def __init__(
        self,
        id: str,
        log_time: str,
        attack_type: ReportAttackType,
        description: str | None,
        parameters: Dict[str, ReportParameterInstance],
        instances: Dict[str, ReportInstance],
        global_metrics: Dict[str, ReportResult]
    ) -> None:
        """
        Initialise a report experiment.

        Parameters
        ----------
        id : str
            Unique identifier for the experiment.

        log_time : str
            Timestamp indicating when the experiment was run.

        attack_type : ReportAttackType
            Attack type associated with this experiment.

        description : str or None
            Optional human-readable description of the experiment.

        parameters : dict[str, ReportParameterInstance]
            Mapping of parameter names to their instantiated values.

        instances : dict[str, ReportInstance]
            Mapping of instance IDs to per-instance results.

        global_metrics : dict[str, ReportResult]
            Mapping of metric names to global (non-aggregated) results.

        """
        self.id: str = id
        self.log_time: str = log_time
        self.attack_type: ReportAttackType = attack_type
        self.description: str | None = description
        self.parameters: Dict[str, ReportParameterInstance] = parameters
        self.instances: Dict[str, ReportInstance] = instances
        self.global_metrics: Dict[str, ReportResult] = global_metrics
        self.aggregate_metrics: Dict[str, ReportResult] = {}

        self.compute_aggregations()

        self.visualisations: List[Visualisation] = self.generate_visualisations(
            [ROCPlot]
        )

    def instance_results(self, result_key: str) -> List:
        """
        Retrieve per-instance metric values for a given metric key.

        Parameters
        ----------
        result_key : str
            Metric name to retrieve from each instance.

        Returns
        -------
        list
            List of metric values (one per instance).
        """
        return [
            instance.results[result_key].value
            for instance in self.instances.values()
        ]

    def compute_aggregations(self) -> None:
        """
        Compute aggregated metrics across instances.

        Aggregations are computed only for metrics whose per-instance
        values are numeric and whose metric definition allows the
        corresponding aggregation type.
        """
        if not self.instances:
            return

        first_instance = next(iter(self.instances.values()))

        for key, metric in self.attack_type.key_metrics.items():
            value = first_instance.results[key].value

            if isinstance(value, float):
                vals = np.asarray(self.instance_results(key), dtype=float)
                result = ReportResult(metric, value={})

                if "mean" in metric.allowed_aggregations:
                    result.value["mean"] = float(vals.mean())

                if "var" in metric.allowed_aggregations:
                    result.value["var"] = float(vals.var())

                if "min" in metric.allowed_aggregations:
                    result.value["min"] = float(vals.min())

                if "max" in metric.allowed_aggregations:
                    result.value["max"] = float(vals.max())

                self.aggregate_metrics[key] = result

    def instance_metric_keys(self) -> List[str]:
        """
        Return the set of metric keys available at the instance level.

        Returns
        -------
        list[str]
            Metric names present in instance results.
        """
        if not self.instances:
            return []

        return list(next(iter(self.instances.values())).results.keys())

    def generate_visualisations(
        self,
        visualisation_classes: Sequence[Type["Visualisation"]]
    ) -> List["Visualisation"]:
        """
        Instantiate visualisations for this experiment.

        Parameters
        ----------
        visualisation_classes : sequence[type[Visualisation]]
            Visualisation classes to attempt to generate.

        Returns
        -------
        list[Visualisation]
            List of generated visualisation objects.
        """
        visualisations: List[Visualisation] = []

        for vis_cls in visualisation_classes:
            vis = vis_cls(self)

            if vis.visualisation_applies_to_experiment():

                visualisations.append(vis)

        return visualisations
    


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
    

class Report:
    """
    Top-level report orchestrator.

    Responsible for:
    - Loading report data (JSON/YAML)
    - Constructing domain objects (metrics, experiments, etc.)
    - Rendering the final report via Jinja templates
    """

    def __init__(
        self,
        report_json: str,
        report_yaml: str
    ) -> None:
        """
        Initialise the report.

        Parameters
        ----------
        report_json : str
            Path to the report JSON file.

        report_yaml : str
            Path to the configuration YAML file.

        """
        self.report_dict: Dict[str, Any] = self.load_report_json(report_json)
        self.report_config: Dict[str, Any] = self.load_report_yaml(report_yaml)

        self.report_md: str | None = None
        self.report_output_folder_name: str | None = None
        self.timestamp: datetime.datetime = datetime.datetime.now()

        self.consume_report_json()
        self.consume_config()

    # ---------------------------------------------------------------------
    # Loaders
    # ---------------------------------------------------------------------

    def load_report_json(self, report_json: str) -> Dict[str, Any]:
        """Load JSON report file."""
        with open(report_json, encoding="utf-8") as f:
            return json.load(f)

    def load_report_yaml(self, report_yaml: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(report_yaml, encoding="utf-8") as f:
            return yaml.safe_load(f)


    # ---------------------------------------------------------------------
    # Processing helpers
    # ---------------------------------------------------------------------

    def process_parameter(self, id: str, data: Dict[str, Any]) -> ReportParameter:
        return ReportParameter(
            name=id,
            label=data["label"],
            description=data["description"],
        )

    def process_attack_category(
        self,
        id: str,
        data: Dict[str, Any],
    ) -> ReportAttackCategory:
        return ReportAttackCategory(
            name=id,
            label=data["label"],
            description=data["description"],
            order=int(data["order"]),
        )

    def process_attack_type(
        self,
        id: str,
        data: Dict[str, Any],
    ) -> ReportAttackType:
        return ReportAttackType(
            name=id,
            label=data["label"],
            description=data["description"],
            category=self.categories.get(data["attack_category"]),
            parameters={
                k: self.process_parameter(k, v)
                for k, v in data["attack_params"].items()
            },
            key_metrics={
                x: self.resolve_metric(x)
                for x in data["key_metrics"]
            },
        )

    def process_parameter_instance(
        self,
        id: str,
        data: Any,
    ) -> ReportParameterInstance:
        return ReportParameterInstance(
            parameter=self.parameters.get(id),
            value=data,
        )

    def process_result(self, id: str, data: Any) -> ReportResult:
        return ReportResult(
            metric=self.resolve_metric(id),
            value=data,
        )

    def process_instance(
        self,
        id: str,
        data: Dict[str, Any],
    ) -> ReportInstance:
        return ReportInstance(
            id=id,
            number=int(id.split("_")[-1]),
            results={
                k: self.process_result(k, v)
                for k, v in data.items()
            },
        )

    def process_experiment(
        self,
        id: str,
        data: Dict[str, Any],
    ) -> ReportExperiment:
        return ReportExperiment(
            id=id,
            log_time=data["log_time"],
            attack_type=self.attack_types.get(data["metadata"]["attack_name"]),
            description=None,
            parameters={
                k: self.process_parameter_instance(k, v)
                for k, v in data["metadata"]["attack_params"].items()
            },
            instances={
                k: self.process_instance(k, v)
                for k, v in data["attack_experiment_logger"]["attack_instance_logger"].items()
            },
            global_metrics={
                k: self.process_result(k, v)
                for k, v in data["metadata"].get("global_metrics", {}).items()
            }
        )

    def resolve_metric(self, id: str) -> ReportMetric:
        """
        Resolve a metric by ID, including pattern-matched metrics.
        """
        if id in self.metrics:
            return self.metrics[id]

        for entry in self.pattern_metrics:
            if entry["regex"].match(id):
                pm = entry["definition"]

                return ReportMetric(
                    name=id,
                    label=pm["label_template"].format(metric_name=id),
                    description=pm["description"],
                    units=pm.get("units"),
                    higher_is_better=pm.get("higher_is_better"),
                    category=pm.get("category"),
                    typical_range=None,
                    notes=pm.get("notes"),
                    allowed_aggregations=pm["allowed_aggregations"],
                )

        raise KeyError(f"Metric '{id}' not found")

    def process_metric(self, id: str, data: Dict[str, Any]) -> ReportMetric:
        return ReportMetric(
            name=id,
            label=data["label"],
            description=data["description"],
            units=data["units"],
            higher_is_better=data["higher_is_better"],
            category=data["category"],
            typical_range=data["typical_range"],
            notes=data["notes"],
            allowed_aggregations=data["allowed_aggregations"],
        )

    # ---------------------------------------------------------------------
    # Build model
    # ---------------------------------------------------------------------

    def consume_report_json(self) -> None:
        """Construct all domain objects from raw JSON."""
        self.report_schema_version = self.report_dict["report_schema_version"]

        self.metrics = {
            k: self.process_metric(k, v)
            for k, v in self.report_dict["metric_catalog"]["metrics"].items()
        }

        self.pattern_metrics = [
            {"regex": re.compile(pm["pattern"]), "definition": pm}
            for pm in self.report_dict["metric_catalog"].get("pattern_metrics", [])
        ]

        self.parameters = {
            k: self.process_parameter(k, v)
            for k, v in self.report_dict["parameter_catalog"]["parameters"].items()
        }

        self.categories = {
            k: self.process_attack_category(k, v)
            for k, v in self.report_dict["attack_category_catalog"]["categories"].items()
        }

        self.attack_types = {
            k: self.process_attack_type(k, v)
            for k, v in self.report_dict["attack_catalog"]["attacks"].items()
        }

        self.experiments = {
            k: self.process_experiment(k, v)
            for k, v in self.report_dict["attacks"].items()
        }

    def consume_config(self) -> None:
        """Load report-level metadata."""
        self.author: str = self.report_config["author"]
        self.project_name: str = self.report_config["project_name"]
        self.project_blurb: str = self.report_config.get(
            "project_blurb",
            "*Add some explanation or background to introduce the project*",
        )
        self.recommendations: str = self.report_config.get(
            "recommendations",
            "*Add user recommendations here*",
        )

    # ---------------------------------------------------------------------
    # Render visualisations
    # ---------------------------------------------------------------------

    def render_visualisations(self, output_dir: str) -> None:
        """
        Render and save visualisation plots for all experiments.

        This method iterates over all experiments associated with the report,
        and for each experiment, calls the ``plot`` method on its registered
        visualisations. Each plot is saved to the ``figures`` subdirectory
        within the specified output directory.

        Parameters
        ----------
        output_dir : str
            Root directory where visualisation images should be written.
            A ``figures`` subdirectory will be created within this path
            if it does not already exist.

        Returns
        -------
        None

        Notes
        -----
        - Assumes that each experiment already has a populated list of
        visualisation objects.

        """
        for experiment in self.experiments.values():
            for vis in experiment.visualisations:
                vis.plot(output_dir=os.path.join(output_dir, "figures"))



    # ---------------------------------------------------------------------
    # Jinja rendering
    # ---------------------------------------------------------------------

    def create_jinja_env(self) -> Environment:
        """Create Jinja environment."""

        templates_dir = resources.files("sacroml.reporting") / "templates"
        
        return Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render_report(
        self,
        template_name: str,
        output_dir: str,
    ) -> str:
        """
        Render the report using a Jinja template.

        Parameters
        ----------
        template_name : str
            Name of the template file.

        output_dir : str
            Output directory.

        Returns
        -------
        str
            Rendered report content.
        """
        env = self.create_jinja_env()
        template = env.get_template(template_name)

        return template.render(
            report=self,
            output_dir=output_dir,
        )