"""TRE SCRIPT FOR USER STORY 1.

This file contains the code needed to run user story 1.

To run: change the user_story key inside the .yaml config file to '1', and run
the 'generate_disclosure_risk_report.py' file.

NOTE: you should not need to change this file at all, set all parameters via
the .yaml file.
"""

import argparse
import os

import yaml

from aisdc.attacks.attack_report_formatter import (  # pylint: disable=import-error
    GenerateTextReport,
)


def generate_report(directory, attack_results, target, outfile):
    """Generate report based on target model."""
    print()
    print("Acting as TRE...")
    print()

    text_report = GenerateTextReport()

    attack_pathname = os.path.join(directory, attack_results)
    text_report.process_attack_target_json(
        attack_pathname, target_filename=os.path.join(directory, target)
    )

    out_pathname = os.path.join(directory, outfile)
    text_report.export_to_file(output_filename=out_pathname, move_files=True)

    print("Results written to " + out_pathname)


def run_user_story(release_config: dict):
    """Run the user story, parsing arguments and then invoking report generation."""
    generate_report(
        release_config["training_artefacts_dir"],
        release_config["attack_results"],
        release_config["target_results"],
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
