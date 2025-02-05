"""TRE SCRIPT FOR USER STORY 2.

This file contains the code needed to run user story 2.

To run: change the user_story key inside the .yaml config file to '2', and run
the 'generate_disclosure_risk_report.py' file.

NOTE: you should not need to change this file at all, set all parameters via
the .yaml file.
"""

import argparse
import importlib
import os
import pickle

import numpy as np
import pandas as pd
import yaml

from sacroml.attacks.attack_report_formatter import GenerateTextReport
from sacroml.attacks.target import Target


def process_dataset(filename, function_name, data_to_be_processed):
    """Process dataset.

    DO NOT CHANGE: this is a wrapper function that allows a callable function
    to be read from a file.
    """
    spec = importlib.util.spec_from_file_location(function_name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, function_name)
    return function(data_to_be_processed)


def generate_report(
    data_processing_filename,
    data_processing_function_name,
    dataset_filename,
    directory,
    target_model,
    attack_results,
    target_filename,
    outfile,
):
    """Generate report based on target model."""
    print()
    print("Acting as TRE...")
    print(
        "(when instructions on how to recreate the dataset have been provided by the researcher)"
    )
    print(directory)
    print()

    # Read in the model supplied by the researcher
    filename = os.path.join(directory, target_model)
    print("Reading target model from " + filename)
    with open(filename, "rb") as f:
        target_model = pickle.load(f)

    # Read the data used by the researcher, and process it using their defined function
    print("Reading data from " + dataset_filename)
    data = pd.read_csv(dataset_filename)

    returned = process_dataset(
        data_processing_filename, data_processing_function_name, data
    )
    X_transformed = returned["X_transformed"]
    y_transformed = returned["y_transformed"]
    train_indices = set(returned["train_indices"])

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for i, label in enumerate(y_transformed):
        if i in train_indices:
            X_train.append(X_transformed[i])
            y_train.append(label)
        else:
            X_test.append(X_transformed[i])
            y_test.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Wrap the model and data in a Target object
    target = Target(model=target_model)
    target.add_processed_data(X_train, y_train, X_test, y_test)

    # TRE calls request_release()
    print("===> now running attacks implicitly via request_release()")
    target_model.request_release(path=directory, ext="pkl", target=target)

    print(f"Please see the files generated in: {directory}")

    # Generate a report indicating calculated disclosure risk
    text_report = GenerateTextReport()
    text_report.process_attack_target_json(
        os.path.join(directory, attack_results),
        target_filename=os.path.join(directory, target_filename),
    )

    text_report.export_to_file(
        output_filename=os.path.join(directory, outfile), move_files=True
    )

    print("Results written to " + str(os.path.join(directory, outfile)))


def run_user_story(release_config: dict):
    """Run the user story, parsing arguments and then invoking report generation."""
    generate_report(
        release_config["data_processing_filename"],
        release_config["data_processing_function_name"],
        release_config["dataset_filename"],
        release_config["training_artefacts_dir"],
        release_config["target_model"],
        release_config["attack_results"],
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
