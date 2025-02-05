"""TRE SCRIPT FOR USER STORY 4.

This file contains the code needed to run user story 4.

To run: change the user_story key inside the .yaml config file to '4', and run
the 'generate_disclosure_risk_report.py' file.

NOTE: you should not need to change this file at all, set all parameters via
the .yaml file.
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml

from sacroml.attacks.target import Target
from sacroml.attacks.worst_case_attack import WorstCaseAttack


def generate_report(
    directory,
    train_probabilities,
    test_probabilities,
):
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

    # Create Target
    target = Target(
        proba_train=np.array(output_train),
        proba_test=np.array(output_test),
        y_train=train_proba["true_label"],
        y_test=test_proba["true_label"],
    )

    # Run the attack
    wca = WorstCaseAttack(n_dummy_reps=10, output_dir=directory)
    wca.attack(target)

    print(f"Results written to {directory}")


def run_user_story(release_config: dict):
    """Run the user story, parsing arguments and then invoking report generation."""
    generate_report(
        release_config["training_artefacts_dir"],
        release_config["train_probabilities"],
        release_config["test_probabilities"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate a risk report after request_release() "
            "has been called by researcher"
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
            f"Invalid command. Try --help to get more detailserror message is {error}"
        )

    run_user_story(config)
