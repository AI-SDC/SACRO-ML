"""Code for automatic report generation."""


from __future__ import annotations

import datetime
import json
import os
import re
from typing import Any, Dict

import yaml
from jinja2 import Environment, FileSystemLoader
from importlib import resources

from sacroml.reporting.models import (
    ReportMetric,
    ReportResult,
    ReportParameter,
    ReportParameterInstance,
    ReportInstance,
    ReportAttackCategory,
    ReportAttackType,
    ReportExperiment,
)



from sacroml.reporting.visualisations import (
    ROCPlot,
    QuantitativeRiskPlot,
    CategoricalRiskPlot,
    CategoricalFractionPlot,
)

NON_METRIC_FIELDS = {"details"}

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

        results = {}

        for k, v in data.items():
            if k in NON_METRIC_FIELDS:
                continue

            results[k] = self.process_result(k, v)

        return ReportInstance(
            id=id,
            number=int(id.split("_")[-1]),
            results=results,
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
                if k not in NON_METRIC_FIELDS
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


        self.experiments = {}

        for exp_id, exp_data in self.report_dict["attacks"].items():
            attack_name = exp_data["metadata"].get("attack_name", "")

            if attack_name == "Attribute inference attack":
                print(f"Skipping unsupported attack: {exp_id}")
                continue

            self.experiments[exp_id] = self.process_experiment(exp_id, exp_data)


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
        Generate and save visualisation plots for all experiments.

        This method instantiates the configured visualisation classes for each
        experiment, filters them based on applicability, and then generates
        the corresponding plots. Generated visualisations are stored on each
        experiment instance and written to disk.

        Parameters
        ----------
        output_dir : str
            Root directory where visualisation images should be written.
            A ``figures`` subdirectory will be created within this path.

        Returns
        -------
        None
        """
        figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        visualisation_classes = [ROCPlot, QuantitativeRiskPlot, CategoricalRiskPlot, CategoricalFractionPlot]

        for experiment in self.experiments.values():
            experiment.visualisations = []

            for vis_cls in visualisation_classes:
                vis = vis_cls(experiment)

                if vis.visualisation_applies_to_experiment():
                    vis.plot(output_dir=figures_dir)
                    experiment.visualisations.append(vis)




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