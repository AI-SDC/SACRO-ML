"""Test report generation."""

from __future__ import annotations

import json
import os
import unittest

import pytest

from aisdc.attacks.attack_report_formatter import (
    AnalysisModule,
    FinalRecommendationModule,
    GenerateJSONModule,
    GenerateTextReport,
    LogLogROCModule,
    SummariseAUCPvalsModule,
    SummariseFDIFPvalsModule,
    SummariseUnivariateMetricsModule,
)


def get_test_report():
    """Create a mock attack result dictionary for use with tests."""
    json_formatted = {
        "log_id": 1024,
        "metadata": {"attack": "WorstCase"},
        "model_params": {"min_samples_leaf": 10},
        "model_name": "RandomForestClassifier",
        "WorstCaseAttack": {"attack_experiment_logger": {"attack_instance_logger": {}}},
    }

    metrics_dict = {
        "P_HIGHER_AUC": 0.5,
        "AUC": 0.6,
        "ACC": 0.6,
        "FDIF01": 0.2,
        "PDIF01": 1.0,
        "FPR": 1.0,
        "TPR": 0.1,
        "fpr": [0.0, 0.0, 1.0],
        "tpr": [0.0, 1.0, 1.0],
    }

    for i in range(10):
        json_formatted["WorstCaseAttack"]["attack_experiment_logger"][
            "attack_instance_logger"
        ]["instance_" + str(i)] = metrics_dict

    return json_formatted


def get_target_report():
    """Create a mock target model dictionary for use with tests."""
    target_formatted = {
        "data_name": "",
        "n_samples": 12960,
        "features": {},
        "n_features": 0,
        "n_samples_orig": 0,
        "model_path": "model.pkl",
        "model_name": "SVC",
        "model_params": {"C": 1.0},
    }

    return target_formatted


class TestGenerateReport(unittest.TestCase):
    """Class which tests the attack_report_formatter.py file."""

    def process_json_from_file(self, json_formatted):
        """Function which handles file input/output from the process_json function."""
        filename = "test.json"
        output_filename = "results.txt"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(json_formatted, f)

        g = GenerateTextReport()
        g.process_attack_target_json(filename)
        g.export_to_file(output_filename)

        with open(output_filename, encoding="utf-8") as file:
            data = file.read()

        return data

    def test_not_implemented(self):
        """Test to make sure analysis module fails expectedly when functions are called directly."""
        a = AnalysisModule()
        with pytest.raises(NotImplementedError):
            a.process_dict()

        with pytest.raises(NotImplementedError):
            str(a)

    def test_json_formatter(self):
        """Test which tests the GenerateJSONModule."""
        g = GenerateJSONModule()
        filename = g.get_output_filename()
        assert filename is not None

        test_filename = "example_filename.json"
        g = GenerateJSONModule(test_filename)
        assert test_filename == g.get_output_filename()

        # check file is overwritten when the same file is passed
        test_filename = "filename_to_rewrite.json"
        g = GenerateJSONModule(test_filename)
        g.clean_file()

        msg_1 = "this should be included in the file"
        msg_2 = "this should also be included in the file"

        g.add_attack_output('{"test_output":"' + msg_1 + '"}', "FirstTestAttack")
        g.add_attack_output('{"test_output":"' + msg_2 + '"}', "SecondTestAttack")

        with open(test_filename, encoding="utf-8") as f:
            file_contents = json.loads(f.read())

        assert msg_1 in file_contents["FirstTestAttack"]["test_output"]
        assert msg_2 in file_contents["SecondTestAttack"]["test_output"]

    def test_pretty_print(self):
        """Test which tests the pretty_print function with nested dictionaries."""
        example_report = {
            "example_a": "example_value",
            "example_b": {"A": 1.0, "B": 1.0},
        }

        g = GenerateTextReport()
        g.pretty_print(example_report, "Example Title")

    def test_process_attack_target_json(self):
        """Test which tests the process_attack_target_json function."""
        target_report = get_target_report()
        target_json = "target.json"

        with open(target_json, "w", encoding="utf-8") as f:
            json.dump(target_report, f)

        json_formatted = get_test_report()

        attack_json = "test.json"
        output_filename = "attack.txt"

        with open(attack_json, "w", encoding="utf-8") as f:
            json.dump(json_formatted, f)

        g = GenerateTextReport()
        g.process_attack_target_json(attack_json, target_json)
        g.export_to_file(output_filename)

        metrics_dict = {
            "P_HIGHER_AUC": 0.001,
            "AUC": 0.8,
            "ACC": 0.7,
            "FDIF01": 0.2,
            "PDIF01": 1.0,
            "FPR": 1.0,
            "TPR": 0.1,
        }

        for i in range(10):
            json_formatted["WorstCaseAttack"]["attack_experiment_logger"][
                "attack_instance_logger"
            ]["instance_" + str(i)] = metrics_dict

        with open(attack_json, "w", encoding="utf-8") as f:
            json.dump(json_formatted, f)

        g = GenerateTextReport()
        g.process_attack_target_json(attack_json, target_json)
        g.export_to_file(output_filename)

    def test_whitespace_in_filenames(self):
        """Test whitespace is removed from the output file when creating a report."""
        json_formatted = get_test_report()

        filename = "test.json"
        output_filename = "filename should be changed.txt"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(json_formatted, f)

        g = GenerateTextReport()
        g.process_attack_target_json(filename)
        g.export_to_file(output_filename)

        assert not os.path.exists("filename should be changed.txt")
        assert os.path.exists("filename_should_be_changed.txt")

    def test_move_files(self):
        """Test the move_files parameter inside export_to_file."""
        filename = "test.json"
        output_filename = "results.txt"
        dummy_model = "dummy_model.txt"

        json_formatted = get_test_report()

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(json_formatted, f)

        with open(dummy_model, "w", encoding="utf-8") as f:
            f.write("dummy_model")

        g = GenerateTextReport()
        g.process_attack_target_json(filename)
        g.export_to_file(output_filename, move_files=True, release_dir="release_dir")

        # Check when no model name has been provided
        assert os.path.exists(os.path.join("release_dir", output_filename))

        g.export_to_file(
            output_filename,
            move_files=True,
            release_dir="release_dir",
            model_filename=dummy_model,
        )

        # Check model file has been copied (NOT moved)
        assert os.path.exists(os.path.join("release_dir", dummy_model))
        assert os.path.exists(dummy_model)

        png_file = "log_roc.png"
        with open(png_file, "w", encoding="utf-8") as f:
            pass

        assert os.path.exists(png_file)

        g = GenerateTextReport()
        g.process_attack_target_json(filename)
        g.export_to_file(
            output_filename,
            move_files=True,
            release_dir="release_dir",
            artefacts_dir="training_artefacts",
        )

        assert not os.path.exists(png_file)
        assert os.path.exists(os.path.join("training_artefacts", png_file))

    def test_complete_runthrough(self):
        """Test process_json file end-to-end when valid parameters are passed."""
        json_formatted = get_test_report()
        _ = self.process_json_from_file(json_formatted)


class TestFinalRecommendationModule(unittest.TestCase):
    """Tests the FinalRecommendatiionModule inside attack_report_formatter.py."""

    def test_instance_based(self):
        """Test process_json function when target model is an instance based model."""
        json_formatted = get_test_report()
        f = FinalRecommendationModule(json_formatted)
        f.process_dict()
        returned = f.get_recommendation()

        immediate_rejection = returned[0]
        assert len(immediate_rejection) == 0

        json_formatted["model_name"] = "SVC"
        f = FinalRecommendationModule(json_formatted)
        f.process_dict()
        returned = f.get_recommendation()

        immediate_rejection = returned[0]
        assert "Model is SVM" in immediate_rejection

        json_formatted["model_name"] = "KNeighborsClassifier"
        f = FinalRecommendationModule(json_formatted)
        f.process_dict()
        returned = f.get_recommendation()

        immediate_rejection = returned[0]
        assert "Model is kNN" in immediate_rejection

    def test_min_samples_leaf(self):
        """Test process_json when the target model includes decision trees."""

        # test when min_samples_leaf > 5
        json_formatted = get_test_report()

        f = FinalRecommendationModule(json_formatted)
        f.process_dict()
        returned = f.get_recommendation()

        immediate_rejection = returned[0]
        assert len(immediate_rejection) == 0

        # test when min_samples_leaf < 5
        json_formatted["model_params"]["min_samples_leaf"] = 2

        f = FinalRecommendationModule(json_formatted)
        f.process_dict()
        returned = f.get_recommendation()

        support_rejection = returned[1]
        support_rejection = ", ".join(support_rejection)
        assert "Min samples per leaf" in support_rejection

    def test_statistically_significant(self):
        """Test statistically significant AUC p-values in FinalRecommendationModule."""
        json_formatted = get_test_report()
        json_formatted["WorstCaseAttack"]["attack_experiment_logger"][
            "attack_instance_logger"
        ] = {}

        metrics_dict = {
            "P_HIGHER_AUC": 0.001,
            "AUC": 0.8,
            "ACC": 0.7,
            "FDIF01": 0.2,
            "PDIF01": 1.0,
            "FPR": 1.0,
            "TPR": 0.1,
        }

        for i in range(10):
            json_formatted["WorstCaseAttack"]["attack_experiment_logger"][
                "attack_instance_logger"
            ]["instance_" + str(i)] = metrics_dict

        f = FinalRecommendationModule(json_formatted)
        f.process_dict()
        returned = f.get_recommendation()

        support_rejection = returned[1]
        support_rejection = ", ".join(support_rejection)

        assert ">10% AUC are statistically significant" in support_rejection
        assert "Attack AUC > threshold" in support_rejection

        metrics_dict["AUC"] = 0.5

        for i in range(10):
            json_formatted["WorstCaseAttack"]["attack_experiment_logger"][
                "attack_instance_logger"
            ]["instance_" + str(i)] = metrics_dict

        f = FinalRecommendationModule(json_formatted)
        f.process_dict()
        returned = f.get_recommendation()

        support_release = returned[2]
        support_release = ", ".join(support_release)
        assert "Attack AUC <= threshold" in support_release

    def test_print(self):
        """Test the FinalRecommendationModule printing."""
        json_formatted = get_test_report()
        f = FinalRecommendationModule(json_formatted)
        assert str(f) == "Final Recommendation"


class TestSummariseUnivariateMetricsModule(unittest.TestCase):
    """Tests the SummariseUnivariateMetricsModule inside attack_report_formatter.py."""

    def test_univariate_metrics_module(self):
        """Test the SummariseUnivariateMetricsModule."""
        json_formatted = get_test_report()

        auc_value = 0.8
        acc_value = 0.7
        fdif01_value = 0.2

        metrics_dict = {
            "P_HIGHER_AUC": 0.001,
            "AUC": auc_value,
            "ACC": acc_value,
            "FDIF01": fdif01_value,
            "PDIF01": 1.0,
            "FPR": 1.0,
            "TPR": 0.1,
        }

        for i in range(10):
            json_formatted["WorstCaseAttack"]["attack_experiment_logger"][
                "attack_instance_logger"
            ]["instance_" + str(i)] = metrics_dict

        f = SummariseUnivariateMetricsModule(json_formatted)
        returned = f.process_dict()

        wca_auc = returned["WorstCaseAttack"]["AUC"]
        for k in wca_auc.keys():
            assert auc_value == pytest.approx(wca_auc[k])

        wca_acc = returned["WorstCaseAttack"]["ACC"]
        for k in wca_acc.keys():
            assert acc_value == pytest.approx(wca_acc[k])

        wca_fdif = returned["WorstCaseAttack"]["FDIF01"]
        for k in wca_fdif.keys():
            assert fdif01_value == pytest.approx(wca_fdif[k])

        assert str(f) == "Summary of Univarite Metrics"

    def test_print(self):
        """Test the SummariseUnivariateMetricsModule printing."""
        json_formatted = get_test_report()
        f = SummariseUnivariateMetricsModule(json_formatted)
        assert str(f) == "Summary of Univarite Metrics"


class TestSummariseAUCPvalsModule(unittest.TestCase):
    """Tests the SummariseAUCPvalsModule inside attack_report_formatter.py."""

    def test_auc_pvals_module(self):
        """Test the SummariseAUCPvalsModule."""
        json_formatted = get_test_report()
        f = SummariseAUCPvalsModule(json_formatted)
        _ = str(f)
        returned = f.process_dict()

        # test the default correction
        assert returned["correction"] == "bh"
        assert returned["n_total"] == 10

        metrics_dict = {"P_HIGHER_AUC": 0.001}

        for i in range(11):
            json_formatted["WorstCaseAttack"]["attack_experiment_logger"][
                "attack_instance_logger"
            ]["instance_" + str(i)] = metrics_dict

        f = SummariseAUCPvalsModule(json_formatted)
        _ = str(f)
        returned = f.process_dict()
        assert returned["n_total"] == 11

        f = SummariseAUCPvalsModule(json_formatted, correction="bo")
        returned = f.process_dict()
        assert returned["correction"] == "bo"

        f = SummariseAUCPvalsModule(json_formatted, correction="xyzabcd")
        with pytest.raises(NotImplementedError):
            _ = f.process_dict()

        _ = json_formatted["WorstCaseAttack"].pop("attack_experiment_logger")
        f = SummariseAUCPvalsModule(json_formatted)

    def test_print(self):
        """Test the SummariseAUCPvalsModule printing."""
        json_formatted = get_test_report()
        f = SummariseAUCPvalsModule(json_formatted)
        assert "Summary of AUC p-values" in str(f)


class TestSummariseFDIFPvalsModule(unittest.TestCase):
    """Test the SummariseFDIFPvalsModule inside attack_report_formatter.py."""

    def test_fdif_pvals_module(self):
        """Test the SummariseFDIFPvalsModule."""
        json_formatted = get_test_report()
        f = SummariseFDIFPvalsModule(json_formatted)
        returned = f.process_dict()

        assert returned["correction"] == "bh"
        assert returned["n_total"] == 10

        returned = f.get_metric_list(
            json_formatted["WorstCaseAttack"]["attack_experiment_logger"]
        )

        assert len(returned) == 10

    def test_print(self):
        """Test the SummariseFDIFPvalsModule printing."""
        json_formatted = get_test_report()
        f = SummariseFDIFPvalsModule(json_formatted)
        assert "Summary of FDIF p-values" in str(f)


class TestLogLogROCModule(unittest.TestCase):
    """Test the LogLogROCModule inside attack_report_formatter.py."""

    def test_loglog_roc_module(self):
        """Test the LogLogROCModule."""
        json_formatted = get_test_report()
        f = LogLogROCModule(json_formatted)
        returned = f.process_dict()

        output_file = (
            f"{json_formatted['log_id']}-{json_formatted['metadata']['attack']}.png"
        )
        assert output_file in returned
        assert os.path.exists(output_file)

        f = LogLogROCModule(json_formatted, output_folder=".")
        returned = f.process_dict()

        output_file = (
            f"{json_formatted['log_id']}-{json_formatted['metadata']['attack']}.png"
        )
        assert output_file in returned
        assert os.path.exists(output_file)

    def test_loglog_multiple_files(self):
        """Test the LogLogROCModule with multiple tests."""
        out_json = get_test_report()
        out_json_copy = get_test_report()
        out_json_copy["log_id"] = 2048

        out_json.update(out_json_copy)

        f = LogLogROCModule(out_json, output_folder=".")
        returned = f.process_dict()

        output_file_1 = f"{out_json['log_id']}-{out_json['metadata']['attack']}.png"
        output_file_2 = (
            f"{out_json_copy['log_id']}-{out_json_copy['metadata']['attack']}.png"
        )

        assert output_file_1 in returned
        assert output_file_2 in returned

    def test_print(self):
        """Test the LogLogROCModule printing."""
        json_formatted = get_test_report()
        f = LogLogROCModule(json_formatted)
        assert str(f) == "ROC Log Plot"
