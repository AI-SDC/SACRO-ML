"""
User story 7 as TRE.

Details can be found here:
https://github.com/AI-SDC/AI-SDC/issues/141

Running
-------

Invoke this code from the root AI-SDC folder with
python -m example_notebooks.user_stories.user_story_7.user_story_7_tre
"""

import argparse
import os
import pickle
import pathlib
import yaml

def generate_report(directory, target_model_filepath):
    """Main method to parse arguments and then invoke report generation."""
    print()
    print("Acting as TRE...")
    print(
        "(when researcher has provided NO INSTRUCTIONS on how to recreate the dataset)"
    )
    print()

    filename = os.path.join(directory, target_model_filepath)
    print("Reading target model from " + filename)
    with open(filename, "rb") as f:
        _ = pickle.load(f)

    print("Attacks cannot be run since the original dataset cannot be recreated")
    print("AISDC cannot provide any help to TRE")


def main():
    """Main method to parse arguments and then invoke report generation."""
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
        help = (
            "Name of yaml configuration file"
        )
    )

    args = parser.parse_args()

    try:
        with open(args.config_file, encoding="utf-8") as handle:
            config = yaml.load(handle, Loader=yaml.loader.SafeLoader)
    except AttributeError as error:  # pragma:no cover
        print("Invalid command. Try --help to get more details" f"error message is {error}")

    generate_report(config['training_artefacts_dir'], config['target_model'])

if __name__ == "__main__":  # pragma:no cover
    main()
