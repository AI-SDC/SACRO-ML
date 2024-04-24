"""
An entry point to run multiple attacks including MIA (worst-case and LIRA)
and attribute inference attack using a single configuration file
with multiple attack configuration.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from typing import Any

from aisdc.attacks.attack import Attack
from aisdc.attacks.attribute_attack import AttributeAttack
from aisdc.attacks.likelihood_attack import LIRAAttack
from aisdc.attacks.target import Target
from aisdc.attacks.worst_case_attack import WorstCaseAttack


class MultipleAttacks(Attack):
    """Class to wrap the MIA and AIA attack codes."""

    def __init__(
        self,
        config_filename: str = None,
    ) -> None:
        super().__init__()
        self.config_filename = config_filename
        """Constructs an object to execute a worst case attack.

        Parameters
        ----------
        config_filename : str
            name of the configuration file which has configurations in a single JSON file
            to support running multiple attacks
        """

    def __str__(self):
        return "Multiple Attacks (MIA and AIA) given configurations"

    def attack(self, target: Target) -> None:
        """
        Runs attacks from a Target object and a target model.

        Parameters
        ----------
        target : attacks.target.Target
            target as an instance of the Target class. Needs to have x_train,
            x_test, y_train and y_test set.
        """
        logger = logging.getLogger("attack-multiple attacks")
        logger.info("Running attacks")
        file_contents = ""
        with open(self.config_filename, "r+", encoding="utf-8") as f:
            file_contents = f.read()

        if file_contents != "":
            config_file_data = json.loads(file_contents)
            for config_obj in config_file_data:
                params = config_file_data[config_obj]
                attack_name = config_obj.split("-")[0]
                attack_obj = None
                if attack_name == "worst_case":
                    attack_obj = WorstCaseAttack(**params)
                elif attack_name == "lira":
                    attack_obj = LIRAAttack(**params)
                elif attack_name == "attribute":
                    attack_obj = AttributeAttack(**params)
                else:
                    attack_names = "'worst_case', 'lira' and 'attribute'"
                    logger.error(
                        """attack name is %s whereas supported attack names are %s: """,
                        attack_name,
                        attack_names,
                    )

                if attack_obj is not None:
                    attack_obj.attack(target)

                if attack_obj is not None:
                    _ = attack_obj.make_report()
        logger.info("Finished running attacks")


class ConfigFile:
    """Module that creates a single JSON configuration file."""

    def __init__(
        self,
        filename: str = None,
    ) -> None:
        self.filename = filename

        dirname = os.path.normpath(os.path.dirname(self.filename))
        os.makedirs(dirname, exist_ok=True)
        # if file doesn't exist, create it
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write("")

    def add_config(self, config_obj: Any, config_attack_type: str) -> None:
        """Add a section of JSON to the file which is already open."""

        # Read the contents of the file and then clear the file
        config_file_data = self.read_config_file()

        # Add the new JSON to the JSON that was in the file, and re-write
        with open(self.filename, "w", encoding="utf-8") as f:
            class_name = config_attack_type + "-" + str(uuid.uuid4())

            if isinstance(config_obj, dict):
                config_file_data[class_name] = config_obj
            elif isinstance(config_obj, str):
                with open(str(config_obj), encoding="utf-8") as fr:
                    config_file_data[class_name] = json.loads(fr.read())

            f.write(json.dumps(config_file_data))

    def read_config_file(self) -> dict:
        """Reads a JSON configuration file and returns dictionary
        with a number of configuration objects.
        """
        with open(self.filename, encoding="utf-8") as f:
            file_contents = f.read()
            if file_contents != "":
                config_file_data = json.loads(file_contents)
            else:
                config_file_data = {}
        return config_file_data


def _run_attack_from_configfile(args):
    """Run a command line attack based on saved files described in .json file."""
    attack_obj = MultipleAttacks(
        config_filename=str(args.config_filename),
    )
    target = Target()
    target.load(args.target_path)
    attack_obj.attack(target)


def main():
    """Main method to parse args and invoke relevant code."""
    parser = argparse.ArgumentParser(add_help=False)

    subparsers = parser.add_subparsers()
    attack_parser_config = subparsers.add_parser("run-attack-from-configfile")
    attack_parser_config.add_argument(
        "-j",
        "--attack-config-json-file-name",
        action="store",
        required=True,
        dest="config_filename",
        type=str,
        default="singleconfig.json",
        help=(
            """Name of the .json file containing details for running
            multiple attacks run. Default = %(default)s"""
        ),
    )

    attack_parser_config.add_argument(
        "-t",
        "--attack-target-folder-path",
        action="store",
        required=True,
        dest="target_path",
        type=str,
        default="target",
        help=(
            """Name of the target directory to load the trained target model and the target data.
        Default = %(default)s"""
        ),
    )

    attack_parser_config.set_defaults(func=_run_attack_from_configfile)
    args = parser.parse_args()
    try:
        args.func(args)
    except AttributeError as e:  # pragma:no cover
        print(e)
        print("Invalid command. Try --help to get more details")


if __name__ == "__main__":  # pragma:no cover
    main()
