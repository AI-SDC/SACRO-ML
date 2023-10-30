"""
TRE script to run the code to do the disclosure risk checking for a
machine learning model that has been trained by a researcher.

Researchers should fill out the relevant parameters in the .yaml file, which should be in the same
directory as this file

TREs can change the script that is run using the user_story parameter at the top of the file

To run this code:
    python generate_disclosure_risk_report.py (with the .yaml file in the same directory)

NOTE: you should not need to change this file at all
"""

import argparse

import yaml
from user_story_1 import user_story_1_tre
from user_story_2 import user_story_2_tre
from user_story_3 import user_story_3_tre
from user_story_4 import user_story_4_tre
from user_story_7 import user_story_7_tre
from user_story_8 import user_story_8_tre

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Run user stories code from a config file")
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

    user_story = config["user_story"]
    if user_story == 'UNDEFINED':
        print("User story not selected, please select a user story by referring to user_stories_flow_chart.png and adding the relevant number to the the first line of 'default_config.yaml'")
    elif user_story == 1:
        user_story_1_tre.run_user_story(config)
    elif user_story == 2:
        user_story_2_tre.run_user_story(config)
    elif user_story == 3:
        user_story_3_tre.run_user_story(config)
    elif user_story == 4:
        user_story_4_tre.run_user_story(config)
    elif user_story == 7:
        user_story_7_tre.run_user_story(config)
    elif user_story == 8:
        user_story_8_tre.run_user_story(config)
    else:
        raise NotImplementedError(f"User story {user_story} has not been implemented")
