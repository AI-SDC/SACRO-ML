"""TRE SCRIPT FOR USER STORY 3.

This file contains the code needed to run user story 3.

To run: change the user_story key inside the .yaml config file to '3', and run
the 'generate_disclosure_risk_report.py' file.

NOTE: you should not need to change this file at all, set all parameters via
the .yaml file.
"""

import argparse
import logging
import os
import pickle

import numpy as np
import yaml

from sacroml.attacks.attack_report_formatter import GenerateTextReport
from sacroml.attacks.likelihood_attack import LIRAAttack
from sacroml.attacks.target import Target
from sacroml.attacks.worst_case_attack import WorstCaseAttack


def generate_report(
    directory,
    target_model,
    X_train,
    y_train,
    X_test,
    y_test,
    target_filename,
    outfile,
):
    """Generate report based on target model."""
    print()
    print("Acting as TRE...")
    print()

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Suppress messages from SACRO-ML -- comment out these lines to
    # see all the sacroml logging statements
    logging.getLogger("attack-reps").setLevel(logging.WARNING)
    logging.getLogger("prep-attack-data").setLevel(logging.WARNING)
    logging.getLogger("attack-from-preds").setLevel(logging.WARNING)

    # Read the model to be released as supplied by the researcher
    model_filename = os.path.join(directory, target_model)
    print("Reading target model from " + model_filename)
    with open(model_filename, "rb") as file:
        target_model = pickle.load(file)

    # Read the training/testing data as supplied by the researcher
    print("Reading training/testing data from ./" + directory)
    train_x = np.loadtxt(os.path.join(directory, X_train))
    train_y = np.loadtxt(os.path.join(directory, y_train))
    test_x = np.loadtxt(os.path.join(directory, X_test))
    test_y = np.loadtxt(os.path.join(directory, y_test))

    # Wrap the training and test data into the Target object
    target = Target(
        model=target_model,
        X_train=train_x,
        y_train=train_y,
        X_test=test_x,
        y_test=test_y,
    )
    target.save(os.path.join(directory, "target"))

    # Run the attack
    wca = WorstCaseAttack(n_dummy_reps=10, output_dir=directory)
    wca.attack(target)

    # Run the LiRA attack to test disclosure risk
    lira_attack_obj = LIRAAttack(n_shadow_models=100, output_dir=directory)
    lira_attack_obj.attack(target)

    text_report = GenerateTextReport()
    text_report.process_attack_target_json(
        os.path.join(directory, "report") + ".json",
        target_filename=os.path.join(directory, "target", target_filename),
    )

    text_report.export_to_file(
        output_filename=os.path.join(directory, outfile),
        move_files=True,
        model_filename=model_filename,
    )

    print("Results written to " + os.path.join(directory, outfile))


def run_user_story(release_config: dict):
    """Run the user story, parsing arguments and then invoking report generation."""
    generate_report(
        release_config["training_artefacts_dir"],
        release_config["target_model"],
        release_config["X_train_path"],
        release_config["y_train_path"],
        release_config["X_test_path"],
        release_config["y_test_path"],
        release_config["target_results"],
        release_config["outfile"],
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
