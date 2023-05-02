"""test_generate_report.py
Copyright (C) Jim Smith 2022 <james.smith@uwe.ac.uk>
"""
import json
import os
import unittest

import pytest

from aisdc.generate_report import (
    AnalysisModule,
    FinalRecommendationModule,
    LogLogROCModule,
    SummariseAUCPvalsModule,
    SummariseFDIFPvalsModule,
    SummariseUnivariateMetricsModule,
    process_json,
)


class TestGenerateReport(unittest.TestCase):
    """class which tests the generate_report.py file"""

    def get_test_report(self):
        """create a mock attack result dictionary for use with tests"""
        json_formatted = {
            "log_id": 1024,
            "metadata": {"attack": "WorstCase attack"},
            "model_params": {"min_samples_leaf": 10},
            "attack_experiment_logger": {"attack_instance_logger": {}},
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
            json_formatted["attack_experiment_logger"]["attack_instance_logger"][
                "instance_" + str(i)
            ] = metrics_dict

        return json_formatted

    def process_json_from_file(self, json_formatted):
        """function which handles file input/output from the process_json function"""
        filename = "test.json"
        output_filename = "results.txt"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(json_formatted, f)

        process_json(filename, output_filename)

        with open(output_filename, encoding="utf-8") as file:
            data = file.read()

        return data

    def clean_up(self, name):
        """removes unwanted files or directory"""
        if os.path.exists(name) and os.path.isfile(name):
            os.remove(name)

    def test_not_implemented(self):
        """test to make sure analysis module fails expectedly when functions are called directly"""
        a = AnalysisModule()
        with pytest.raises(NotImplementedError):
            a.process_dict()

        with pytest.raises(NotImplementedError):
            str(a)

    def test_svm(self):
        """test the process_json function when the target model is an SVM"""
        json_formatted = self.get_test_report()

        f = FinalRecommendationModule(json_formatted)
        returned = f.process_dict()

        self.assertEqual(len(returned["score_descriptions"]), 0)

        json_formatted["model"] = "SVC"
        f = FinalRecommendationModule(json_formatted)
        returned = f.process_dict()

        self.assertIn("Model is SVM", returned["score_descriptions"][0])

    def test_min_samples_leaf(self):
        """test the process_json function when the target model is a random forest"""
        json_formatted = self.get_test_report()

        f = FinalRecommendationModule(json_formatted)
        returned = f.process_dict()

        self.assertEqual(len(returned["score_descriptions"]), 0)

        json_formatted["model_params"]["min_samples_leaf"] = 2
        f = FinalRecommendationModule(json_formatted)
        returned = f.process_dict()

        self.assertIn("Min samples per leaf", returned["score_descriptions"][0])

    def test_statistically_significant(self):
        """test the statistically significant AUC p-values check in FinalRecommendationModule"""
        json_formatted = self.get_test_report()
        json_formatted["attack_experiment_logger"]["attack_instance_logger"] = {}

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
            json_formatted["attack_experiment_logger"]["attack_instance_logger"][
                "instance_" + str(i)
            ] = metrics_dict

        f = FinalRecommendationModule(json_formatted)
        returned = f.process_dict()

        self.assertIn(
            ">10% AUC are statistically significant", returned["score_descriptions"][0]
        )
        self.assertIn("Attack AUC > threshold", returned["score_descriptions"][1])

    def test_univariate_metrics_module(self):
        """test the SummariseUnivariateMetricsModule"""
        json_formatted = self.get_test_report()
        f = SummariseUnivariateMetricsModule(json_formatted)
        _ = f.process_dict()

    def test_auc_pvals_module(self):
        """test the SummariseAUCPvalsModule"""
        json_formatted = self.get_test_report()
        f = SummariseAUCPvalsModule(json_formatted)
        _ = f.process_dict()

        f = SummariseAUCPvalsModule(json_formatted, correction="bo")
        _ = f.process_dict()

        with pytest.raises(NotImplementedError):
            f = SummariseAUCPvalsModule(json_formatted, correction="xyzabcd")
            _ = f.process_dict()

    def test_fdif_pvals_module(self):
        """test the SummariseFDIFPvalsModule"""
        json_formatted = self.get_test_report()
        f = SummariseFDIFPvalsModule(json_formatted)
        _ = f.process_dict()
        _ = f.get_metric_list(json_formatted["attack_experiment_logger"])

    def test_loglog_roc_module(self):
        """test the LogLogROCModule"""
        json_formatted = self.get_test_report()
        f = LogLogROCModule(json_formatted)
        _ = f.process_dict()

        f = LogLogROCModule(json_formatted, output_folder="./")
        _ = f.process_dict()

    def test_complete_runthrough(self):
        """test the full process_json file end-to-end when valid parameters are passed"""
        json_formatted = self.get_test_report()
        _ = self.process_json_from_file(json_formatted)

    def test_cleanup(self):
        """gets rid of files created during tests"""
        names = ["test.json", "results.txt", "1024-WorstCase attack.png"]
        for name in names:
            self.clean_up(name)
