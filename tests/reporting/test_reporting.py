"""Test Reporting model objects."""
import re
import json
import yaml
import datetime
import os

from unittest.mock import patch, Mock, call
import pytest
from sacroml.reporting import Report
from types import SimpleNamespace

@pytest.fixture(name="dummy_report_setup")
def fixture_dummy_report_setup():
    def __init__(self):
        pass

    with patch.object(Report, "__init__", __init__):
        dummy = Report()
    return dummy


class TestReporting:
    def test_process_parameter(self, dummy_report_setup):
        report = dummy_report_setup
        parameter = report.process_parameter(
            "n_reps",
            {
                "label": "Number of repetitions",
                "description": "Number of independent repetitions used to estimate variability.",
                "parameter_scope": "attack",
                "value_type": "integer",
                "notes": "",
            },
        )
        assert parameter.name == "n_reps"
        assert parameter.label == "Number of repetitions"
        assert (
            parameter.description
            == "Number of independent repetitions used to estimate variability."
        )

    def test_process_attack_category(self, dummy_report_setup):
        report = dummy_report_setup
        attack_category = report.process_attack_category(
            "structural",
            {
                "label": "Structural attacks",
                "description": "Attacks exploiting deterministic or structural properties of the target model.",
                "order": 10
            }
        
        )

        assert attack_category.name == "structural"
        assert (
            attack_category.label
            == "Structural attacks"
        )
        assert (
            attack_category.description
            == "Attacks exploiting deterministic or structural properties of the target model."
        )
        assert attack_category.order == 10

    def test_process_attack_type(self, dummy_report_setup):
        report = dummy_report_setup

        report.process_parameter = Mock(
            side_effect=lambda k, v: f"param:{k}"
        )

        report.resolve_metric = Mock(
            side_effect=lambda x: f"metric:{x}"
        )

        report.categories = {
            "structural": "structural category"
        }

        attack_type = report.process_attack_type(
            "WorstCase attack",
            {
                "label": "Worst‑Case Membership Inference Attack",
                "description": "Assumes the attacker has maximal auxiliary knowledge.",
                "attack_category": "structural",
                "key_metrics": ["AUC", "TPR"],
                "attack_params": {
                    "n_reps": {
                        "label": "Number of repetitions",
                        "description": "Independent repetitions of the attack.",
                    },
                    "p_thresh": {
                        "label": "Probability threshold",
                        "description": "Decision threshold for membership prediction.",
                    },
                },
            },
        )

        assert attack_type.name == "WorstCase attack"
        assert attack_type.label == "Worst‑Case Membership Inference Attack"
        assert attack_type.description == (
            "Assumes the attacker has maximal auxiliary knowledge."
        )
        assert attack_type.category == "structural category"

        assert attack_type.parameters == {
            "n_reps": "param:n_reps",
            "p_thresh": "param:p_thresh",
        }

        assert attack_type.key_metrics == {
            "AUC": "metric:AUC",
            "TPR": "metric:TPR",
        }

        report.process_parameter.assert_any_call(
            "n_reps",
            {
                "label": "Number of repetitions",
                "description": "Independent repetitions of the attack.",
            },
        )

        report.process_parameter.assert_any_call(
            "p_thresh",
            {
                "label": "Probability threshold",
                "description": "Decision threshold for membership prediction.",
            },
        )

        report.resolve_metric.assert_any_call("AUC")
        report.resolve_metric.assert_any_call("TPR")

    def test_process_parameter_instance(self, dummy_report_setup):
        report = dummy_report_setup
        mock_parameter = Mock(name="parameter")
        report.parameters = {
            "n_shadow_models": mock_parameter
        }
        parameter_instance = report.process_parameter_instance("n_shadow_models", 100)
        assert parameter_instance.parameter is mock_parameter
        assert parameter_instance.value == 100

    def test_process_result(self, dummy_report_setup):
        report = dummy_report_setup
        mock_metric = Mock(name="metric")
        report.metrics = {
            "TPR": mock_metric
        }
        result = report.process_result("TPR", 0.73869347)
        assert result.metric is mock_metric
        assert result.value == 0.73869347

    def test_process_instance(self, dummy_report_setup):
        report = dummy_report_setup
        instance_example_data = {
            "TPR": 0.73869347,
            "FPR": 0.40935673,
            "details": "This is the first instance",
            "fpr": [
              0.0,
              0.08187134502923976,
              0.08771929824561403,
              0.08771929824561403,
              0.0935672514619883,
              0.0935672514619883,
              0.28654970760233917,
              1.0
            ]}
        report.process_result = Mock(
            side_effect=lambda k, v: f"result:{k}"
        )
        instance = report.process_instance("instance_0", instance_example_data)
        assert instance.id == "instance_0"
        assert instance.number == 0
        assert instance.results == {
            "TPR": "result:TPR",
            "FPR": "result:FPR",
            "fpr": "result:fpr"
        }
        report.process_result.assert_any_call(
            "TPR", 0.73869347
        )
        report.process_result.assert_any_call(
            "FPR", 0.40935673
        )
        report.process_result.assert_any_call(
            "fpr", [
                          0.0,
                          0.08187134502923976,
                          0.08771929824561403,
                          0.08771929824561403,
                          0.0935672514619883,
                          0.0935672514619883,
                          0.28654970760233917,
                          1.0
                        ]
        )
        assert call("details", "This is the first instance") not in report.process_result.call_args_list

    def test_process_experiment(self, dummy_report_setup):
        report = dummy_report_setup

        example_experiment = {
            "log_id": "test-log-id",
            "log_time": "06/01/2026 17:04:04",
            "metadata": {
                "attack_name": "LiRA Attack",
                "attack_params": {
                    "n_shadow_models": 100,
                    "p_thresh": 0.05,
                },
                "global_metrics": {
                    "AUC_sig": "Significant at p=0.05",
                },
            },
            "attack_experiment_logger": {
                "attack_instance_logger": {
                    "instance_0": {
                        "AUC": 0.70,
                        "TPR": 0.74,
                        "FPR": 0.41,
                        "fpr": [0.0, 0.5, 1.0],
                        "tpr": [0.0, 0.7, 1.0],
                        "roc_thresh": [1.0, 0.5, 0.0],
                    }
                }
            },
        }
        report.process_parameter_instance = Mock(
                    side_effect=lambda k, v: f"paraminstance:{k}"
                )
        report.process_instance = Mock(
                            side_effect=lambda k, v: f"instance:{k}"
                        )
        report.process_result = Mock(
                            side_effect=lambda k, v: f"result:{k}"
                        )

        mock_attack_type = Mock()
        mock_attack_type.key_metrics = {}
        report.attack_types = {
            "LiRA Attack": mock_attack_type
        }

        experiment = report.process_experiment("LiRA Attack_ac8456b6-57ae-4c68-9a41-a13cb5fff973", example_experiment)

        assert (
            experiment.id
            == "LiRA Attack_ac8456b6-57ae-4c68-9a41-a13cb5fff973"
        )

        assert experiment.log_time == "06/01/2026 17:04:04"

        assert experiment.attack_type is mock_attack_type

        assert experiment.description is None

        assert experiment.parameters == {
            "n_shadow_models": "paraminstance:n_shadow_models",
            "p_thresh": "paraminstance:p_thresh",
        }

        assert experiment.instances == {
            "instance_0": "instance:instance_0",
        }

        assert experiment.global_metrics == {
            "AUC_sig": "result:AUC_sig",
        }
        report.process_parameter_instance.assert_any_call(
            "n_shadow_models",
            100,
        )

        report.process_parameter_instance.assert_any_call(
            "p_thresh",
            0.05,
        )

        assert report.process_parameter_instance.call_count == 2

        report.process_instance.assert_called_once_with(
            "instance_0",
            example_experiment["attack_experiment_logger"][
                "attack_instance_logger"
            ]["instance_0"],
        )

        report.process_result.assert_called_once_with(
            "AUC_sig",
            "Significant at p=0.05",
        )

    def test_resolve_metric_found_in_metrics(self, dummy_report_setup):
        report = dummy_report_setup
        metric = Mock()
        report.metrics = {"AUC": metric}
        result = report.resolve_metric("AUC")
        assert result is metric

    def test_resolve_metric_pattern(self, dummy_report_setup):
        report = dummy_report_setup
        example_pattern = {
        "pattern": "^(FMAX|FMIN|FDIF|PDIF)[0-9]+$",
        "label_template": "{metric_name}",
        "description": "Threshold-based extremum or difference metric computed at a specified operating point.",
        "units": None,
        "higher_is_better": True,
        "category": "threshold_analysis",
        "allowed_aggregations": [
          "mean",
          "min",
          "max"
        ],
        "notes": "Metric name suffix indicates the threshold or operating point."
      }
        report.metrics = {}
        report.pattern_metrics = [{"regex": re.compile(example_pattern["pattern"]), "definition":example_pattern}]
        result = report.resolve_metric("FMAX01")
        assert result.label == "FMAX01"
        assert result.name == "FMAX01"
        assert result.description == "Threshold-based extremum or difference metric computed at a specified operating point."
        assert result.units is None
        assert result.higher_is_better is True
        assert result.category == "threshold_analysis"
        assert result.typical_range is None
        assert result.notes == "Metric name suffix indicates the threshold or operating point."
        assert result.allowed_aggregations == ["mean","min","max"]

    def test_resolve_metric_not_found(self, dummy_report_setup):
        report = dummy_report_setup

        report.metrics = {}
        report.pattern_metrics = {}

        with pytest.raises(KeyError, match="Metric 'BOB' not found"):
            report.resolve_metric("BOB")

    def test_process_metric(self, dummy_report_setup):
        report = dummy_report_setup

        result = report.process_metric(
            "TPR",
            {
                "label": "True Positive Rate",
                "description": "Fraction of true members correctly identified.",
                "units": None,
                "higher_is_better": True,
                "category": "performance",
                "typical_range": [0.0,1.0],
                "allowed_aggregations": [
                "mean",
                "min",
                "max",
                "stddev",
                "var"
                ],
                "notes": None
            }
        )

        assert result.name == "TPR"
        assert result.label == "True Positive Rate"
        assert result.description == "Fraction of true members correctly identified."
        assert result.units is None
        assert result.category == "performance"
        assert result.typical_range == [0.0, 1.0]
        assert result.allowed_aggregations == ["mean", "min", "max", "stddev", "var"]
        assert result.notes is None

    def test_load_report_json(self, dummy_report_setup, tmp_path):
        report = dummy_report_setup

        expected = {
                "label": "True Positive Rate",
                "description": "Fraction of true members correctly identified.",
                "units": None,
                "higher_is_better": True,
                "category": "performance",
                "typical_range": [0.0,1.0],
                "allowed_aggregations": [
                "mean",
                "min",
                "max",
                "stddev",
                "var"
                ],
                "notes": None
            }

        json_file = tmp_path / "report.json"

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(expected, f)

        result = report.load_report_json(str(json_file))

        assert result == expected

    def test_load_report_yaml(self, dummy_report_setup, tmp_path):
        report = dummy_report_setup

        expected = {
                "author": "Hannah Gordon",
                "description": "Test configuration"
            }

        yaml_file = tmp_path / "report.yaml"

        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(expected, f)

        result = report.load_report_yaml(str(yaml_file))

        assert result == expected

    def test_report_init(self):
        with (
            patch.object(Report, "load_report_json", return_value={"test_json": "json_data"}) as mock_load_report_json,
            patch.object(Report, "load_report_yaml", return_value={"test_yaml": "yaml_data"}) as mock_load_report_yaml,
            patch.object(Report, "consume_report_json", return_value={}) as mock_consume_report_json,
            patch.object(Report, "consume_config", return_value={}) as mock_consume_config
        ):
            before = datetime.datetime.now()
            report = Report("example.json", "example.yaml")
            after = datetime.datetime.now()

            mock_load_report_json.assert_called_once_with("example.json")
            mock_load_report_yaml.assert_called_once_with("example.yaml")
            assert report.report_dict == {"test_json": "json_data"}
            assert report.report_config == {"test_yaml": "yaml_data"}
            mock_consume_report_json.assert_called_once()
            mock_consume_config.assert_called_once()

            assert report.report_md is None
            assert report.report_output_folder_name is None
            assert before <= report.timestamp <= after

    def test_consume_report_json(self, dummy_report_setup):
        report = dummy_report_setup
        report.report_dict = {
        "report_schema_version": "1.0",

        "metric_catalog": {
            "metrics": {
                "AUC": {},
            },
            "pattern_metrics": [
                {
                    "pattern": r"^TPR@.+$",
                    "label_template": "{metric_name}",
                }
            ],
        },

        "parameter_catalog": {
            "parameters": {
                "p_thresh": {},
            }
        },

        "attack_category_catalog": {
            "categories": {
                "structural": {},
            }
        },

        "attack_catalog": {
            "attacks": {
                "LiRA Attack": {},
            }
        },

        "attacks": {
            "exp_1": {
                "metadata": {
                    "attack_name": "LiRA Attack",
                }
            },
            "exp_2": {
                "metadata": {
                    "attack_name": "Attribute inference attack",
                }
            },
        },
        }

        report.process_metric = Mock(side_effect=lambda k, v: f"metric:{k}")
        report.process_parameter = Mock(side_effect=lambda k, v: f"parameter:{k}")
        report.process_attack_category = Mock(side_effect=lambda k, v: f"category:{k}")
        report.process_attack_type = Mock(side_effect=lambda k, v: f"attack:{k}")
        report.process_experiment = Mock(side_effect=lambda k, v: f"experiment:{k}")
        report.consume_report_json()
        assert report.report_schema_version == "1.0"
        assert report.metrics == {"AUC": "metric:AUC"}
        assert report.parameters == {"p_thresh": "parameter:p_thresh"}
        assert report.categories == {"structural": "category:structural"}
        assert report.attack_types == {"LiRA Attack": "attack:LiRA Attack"}
        assert report.experiments == {"exp_1": "experiment:exp_1"}
        assert len(report.pattern_metrics) == 1
        assert report.pattern_metrics[0]["definition"]["pattern"] == r"^TPR@.+$"
        assert report.pattern_metrics[0]["regex"].match("TPR@0.1")
        report.process_metric.assert_called_once_with("AUC",{})
        report.process_parameter.assert_called_once_with("p_thresh",{})
        report.process_attack_category.assert_called_once_with("structural",{})
        report.process_attack_type.assert_called_once_with("LiRA Attack",{})
        report.process_experiment.assert_called_once_with("exp_1",report.report_dict["attacks"]["exp_1"])

    def test_consume_config(self, dummy_report_setup):
        report = dummy_report_setup
        report.report_config = {
                "author": "Hannah Gordon",
                "project_name": "Test configuration name",
                "project_blurb": "Test configuration blurb",
                "recommendations": "Test configuration recommendation"
            }
        report.consume_config()
        assert report.author == "Hannah Gordon"
        assert report.project_name == "Test configuration name"
        assert report.project_blurb == "Test configuration blurb"
        assert report.recommendations == "Test configuration recommendation"

    def test_render_visualisations(self, dummy_report_setup):
        report = dummy_report_setup
        figures_dir = os.path.join("output", "figures")
        experiment = SimpleNamespace()
        report.experiments = {"exp1": experiment}
       

        with (
            patch("sacroml.reporting.report.os.makedirs") as mock_makedirs,
            patch("sacroml.reporting.report.ROCPlot") as mock_roc,
        ):
            
            roc_vis = Mock()
            mock_roc.return_value = roc_vis
            roc_vis.visualisation_applies_to_experiment.return_value = True
            report.render_visualisations("output")

            mock_makedirs.assert_called_once_with(figures_dir,exist_ok=True)

            roc_vis.plot.assert_called_once_with(output_dir=figures_dir)
            assert experiment.visualisations == [roc_vis]

    def test_create_jinja_env(self, dummy_report_setup):
        report = dummy_report_setup
        env = report.create_jinja_env()
        assert len(env.list_templates()) > 0

    def test_render_report(self, dummy_report_setup):
        report = dummy_report_setup
        mock_template = Mock()
        mock_template.render.return_value = "<html>report</html>"

        mock_env = Mock()
        mock_env.get_template.return_value = mock_template

        report.create_jinja_env = Mock(return_value=mock_env)

        result = report.render_report(
            "report.html.j2",
            "output",
        )

        report.create_jinja_env.assert_called_once()

        mock_env.get_template.assert_called_once_with(
            "report.html.j2"
        )

        mock_template.render.assert_called_once_with(
            report=report,
            output_dir="output",
        )

        assert result == "<html>report</html>"
