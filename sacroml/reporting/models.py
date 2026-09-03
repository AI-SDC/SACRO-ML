from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

"""
Data models for the sacroml.reporting module.

This module defines the core data structures used to represent
experiment results, including:

- Metrics and metric results
- Parameters and parameter values
- Attack categories and types
- Experiments and their instances

These classes form an in-memory representation of the report input
(JSON/YAML) and provide light processing such as aggregation of
per-instance metrics.

The models are intentionally independent of:
- visualisation logic (plotting)
- report rendering (Jinja templates)
- file I/O

They are used by the reporting pipeline to organise and prepare data
for visualisation and rendering.
"""


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

    A result associates a `ReportMetric` definition with an observed value
    and an optional context describing where or how the metric was measured.

    The value may take one of several forms:

    - A scalar (e.g. float, int, string) for global metrics
    - A dictionary of aggregated statistics (e.g. mean, min, max, var)
    - A list of numeric values (e.g. per-instance values or curve data)
    """

    def __init__(
        self,
        metric: ReportMetric,
        value: Union[Any, Dict[str, float], List[float]],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialise a report result.

        Parameters
        ----------
        metric : ReportMetric
            The metric definition associated with this result.

        value : Any or dict[str, float] or list[float]
            The value of the metric.

        context : dict or None
            Optional contextual information describing where the metric
            applies (e.g. feature name, data split, attribute type).
        """
        self.metric: ReportMetric = metric
        self.value: Union[Any, Dict[str, float], List[float]] = value
        self.context: Dict[str, Any] = context or {}

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

        self.visualisations: List[Visualisation] = []

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

    