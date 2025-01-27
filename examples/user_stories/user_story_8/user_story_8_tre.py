"""TRE SCRIPT FOR USER STORY 8.

This file contains the code needed to run user story 8.

NOTE: this user story will not produce an output, this user story covers cases
where the researcher has not provided enough information. See user stories 1, 2
or 3 for guidance on what you need to release a model.

To run: change the user_story key inside the .yaml config file to '8', and run
the 'generate_disclosure_risk_report.py' file.

NOTE: you should not need to change this file at all, set all parameters via
the .yaml file.
"""

import argparse
import os
import pickle

import yaml


def generate_report(directory, target_model_filepath):
    """Parse arguments and then invoke report generation."""
    print()
    print("Acting as TRE...")
    print(
        "(when researcher has provided NO INSTRUCTIONS on how to recreate the dataset)"
    )
    print()

    filename = os.path.join(directory, target_model_filepath)
    print("Reading target model from " + filename)
    with open(filename, "rb") as file:
        _ = pickle.load(file)

    print("Attacks cannot be run since the original dataset cannot be recreated")
    print("AISDC cannot provide any help to TRE")
    print("In future, the researcher should use the Target.save() function or provide")
    print("copies of their training data")


def run_user_story(release_config: dict):
    """Run the user story, parsing arguments and then invoking report generation."""
    generate_report(
        release_config["training_artefacts_dir"], release_config["target_model"]
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
