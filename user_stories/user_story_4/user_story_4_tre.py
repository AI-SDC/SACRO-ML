"""TRE SCRIPT FOR USER STORY 4.

This file contains the code needed to run user story 4.

To run: change the user_story key inside the .yaml config file to '4', and run
the 'generate_disclosure_risk_report.py' file.

NOTE: you should not need to change this file at all, set all parameters via
the .yaml file.
"""

import argparse
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
import yaml

from aisdc.attacks.attack_report_formatter import (  # pylint: disable=import-error
    GenerateTextReport,
)
from aisdc.attacks.likelihood_attack import LIRAAttack  # pylint: disable=import-error
from aisdc.attacks.target import Target  # pylint: disable=import-error
from aisdc.attacks.worst_case_attack import (  # pylint: disable=import-error
    WorstCaseAttack,
)


def generate_report(
    directory,
    train_probabilities,
    test_probabilities,
    attack_output_name,
    outfile,
):  # pylint: disable=too-many-arguments, disable=too-many-locals
    """Generate report based on target model."""
    print()
    print("Acting as TRE...")
    print()

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Read and process train/test probabilities for disclosure checking
    train_proba = pd.read_csv(train_probabilities)
    test_proba = pd.read_csv(test_probabilities)

    def sort_prob_row(row):
        label = row["true_label"]
        prob = row["probability"]

        if label == 0:
            return prob
        return 1 - prob

    zero_class_train = train_proba.apply(sort_prob_row, axis=1)
    zero_class_test = train_proba.apply(sort_prob_row, axis=1)

    output_train = pd.DataFrame()
    output_train["prob_0"] = zero_class_train
    output_train["prob_1"] = 1 - zero_class_train

    output_test = pd.DataFrame()
    output_test["prob_0"] = zero_class_test
    output_test["prob_1"] = 1 - zero_class_test

    # Run the attack
    wca = WorstCaseAttack(
        n_dummy_reps=10, output_dir=directory, report_name=attack_output_name
    )
    wca.attack_from_preds(
        train_preds=np.array(output_train),
        test_preds=np.array(output_test),
        train_correct=train_proba["true_label"],
        test_correct=test_proba["true_label"],
    )

    # Write results to a file
    _ = wca.make_report()

    text_report = GenerateTextReport()
    text_report.process_attack_target_json(
        os.path.join(directory, attack_output_name) + ".json",
    )

    text_report.export_to_file(
        output_filename=os.path.join(directory, outfile),
        move_files=True,
    )

    print("Results written to " + os.path.join(directory, outfile))


def run_user_story(release_config: dict):
    """Run the user story, parsing arguments and then invoking report generation."""
    generate_report(
        release_config["training_artefacts_dir"],
        release_config["train_probabilities"],
        release_config["test_probabilities"],
        release_config["attack_output_name"],
        release_config["outfile"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate a risk report after request_release() has been called by researcher"
        )
    )

    parser.add_argument(
        "--config_file",
        type=str,
        action="store",
        dest="config_file",
        required=False,
        default="default_config.yaml",
        help=("Name of yaml configuration file"),
    )

    args = parser.parse_args()

    try:
        with open(args.config_file, encoding="utf-8") as handle:
            config = yaml.load(handle, Loader=yaml.loader.SafeLoader)
    except AttributeError as error:
        print(
            "Invalid command. Try --help to get more details"
            f"error message is {error}"
        )

    run_user_story(config)
