"""
TRE SCRIPT FOR USER STORY 2

This file contains the code needed to run user story 2

To run: change the user_story key inside the .yaml config file to '2', and run the 
'generate_disclosure_risk_report.py' file

NOTE: you should not need to change this file at all, set all parameters via the .yaml file

"""
import argparse
import os
import pickle
import importlib

import numpy as np
import pandas as pd
import yaml

from aisdc.attacks.attack_report_formatter import (  # pylint: disable=import-error
    GenerateTextReport,
)
from aisdc.attacks.target import Target  # pylint: disable=import-error

# from .data_processing_researcher import process_dataset

def process_dataset(filename, function_name, data_to_be_processed):
    """ DO NOT CHANGE: a wrapper function that allows a callable function to be read from a file """
    spec = importlib.util.spec_from_file_location(function_name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    function = getattr(module, function_name)
    result = function(data_to_be_processed)
    return result

def generate_report(
    data_processing_filename,
    data_processing_function_name,
    dataset_filename,
    directory,
    target_model,
    attack_results,
    target_filename,
    outfile,
):  # pylint: disable=too-many-locals, disable=too-many-arguments
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

    returned = process_dataset(data_processing_filename, data_processing_function_name, data)
    x_transformed = returned["x_transformed"]
    y_transformed = returned["y_transformed"]
    train_indices = set(returned["train_indices"])

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for i, label in enumerate(y_transformed):
        if i in train_indices:
            x_train.append(x_transformed[i])
            y_train.append(label)
        else:
            x_test.append(x_transformed[i])
            y_test.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Wrap the model and data in a Target object
    target = Target(model=target_model)
    target.add_processed_data(x_train, y_train, x_test, y_test)

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

    text_report.export_to_file(output_filename=os.path.join(directory, outfile), move_files=True)

    print("Results written to " + str(os.path.join(directory, outfile)))

def run_user_story(release_config: dict):
    """Main method to parse arguments and then invoke report generation."""

    generate_report(
        release_config['data_processing_filename'],
        release_config['data_processing_function_name'],
        release_config["dataset_filename"],
        release_config["training_artefacts_dir"],
        release_config["target_model"],
        release_config["attack_results"],
        release_config["target_results"],
        release_config["outfile"],
    )

if __name__ == "__main__":  # pragma:no cover
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
    except AttributeError as error:  # pragma:no cover
        print(
            "Invalid command. Try --help to get more details"
            f"error message is {error}"
        )

    run_user_story(config)
