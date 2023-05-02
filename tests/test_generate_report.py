"""test_worst_case_attack.py
Copyright (C) Jim Smith 2022 <james.smith@uwe.ac.uk>
"""
import json
import os
import pytest

import numpy as np

from aisdc.generate_report import process_json
from aisdc.generate_report import AnalysisModule, FinalRecommendationModule, SummariseUnivariateMetricsModule, SummariseAUCPvalsModule, SummariseFDIFPvalsModule, LogLogROCModule


class TestGenerateReport(unittest.TestCase):
    def get_test_report(self):
        json_formatted = {}
        json_formatted["model_params"] = {}
        json_formatted["model_params"]["min_samples_leaf"] = 10
        json_formatted["attack_experiment_logger"] = {}
        json_formatted["attack_experiment_logger"]["attack_instance_logger"] = {}

        metrics_dict = {
            "P_HIGHER_AUC": 0.5,
            "AUC": 0.6,
            "ACC": 0.6,
            "FDIF01": 0.2,
            "PDIF01": 1.0,
            "FPR": 1.0,
            "TPR": 0.1,
        }

        for i in range(10):
            json_formatted["attack_experiment_logger"]["attack_instance_logger"][
                "instance_" + str(i)
            ] = metrics_dict

        return json_formatted

    def process_json_from_file(self, json_formatted):
        filename = "test.json"
        output_filename = "results.txt"

        with open(filename, "w") as f:
            json.dump(json_formatted, f)

        process_json(filename, output_filename)

        with open(output_filename) as file:
            data = file.read()

        return data

    def clean_up(self, name):
        """removes unwanted files or directory"""
        if os.path.exists(name) and os.path.isfile(name):
            os.remove(name)

    def test_not_implemented(self):
        a = AnalysisModule()
        with pytest.raises(NotImplementedError):
            a.process_dict()

        with pytest.raises(NotImplementedError):
            str(a)

    def test_svm(self):
        json_formatted = self.get_test_report()

        f = FinalRecommendationModule(json_formatted)
        returned = f.process_dict()
    
        self.assertEqual(len(returned['score_descriptions']),0)

        json_formatted['model'] = 'SVC'
        f = FinalRecommendationModule(json_formatted)
        returned = f.process_dict()
        
        self.assertIn('Model is SVM', returned['score_descriptions'][0])

    def test_min_samples_leaf(self):
        json_formatted = self.get_test_report()
        
        f = FinalRecommendationModule(json_formatted)
        returned = f.process_dict()

        self.assertEqual(len(returned['score_descriptions']),0)

        json_formatted['model_params']['min_samples_leaf'] = 2
        f = FinalRecommendationModule(json_formatted)
        returned = f.process_dict()

        self.assertIn('Min samples per leaf',returned['score_descriptions'][0])

    def test_complete_runthrough(self):
        json_formatted = self.get_test_report()
        _ = self.process_json_from_file(json_formatted)

    def test_cleanup(self):
        """gets rid of files created during tests"""
        names = ["test.json", "results.txt"]
        for name in names:
            self.clean_up(name)
