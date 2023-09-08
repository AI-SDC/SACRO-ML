"""
User story 3 as TRE.

Details can be found here:
https://github.com/AI-SDC/AI-SDC/issues/141

Running
-------

Invoke this code from the root AI-SDC folder with
python -m example_notebooks.user_stories.user_story_3.user_story_3_tre
"""

import argparse
import json
import logging
import os
import pickle

import numpy as np
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
    target_model,
    x_train,
    y_train,
    x_test,
    y_test,
    attack_output_name,
    target_filename,
    outfile,
):  # pylint: disable=too-many-arguments, disable=too-many-locals
    """Generate report based on target model."""

    print()
    print("Acting as TRE...")
    print()

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Suppress messages from AI-SDC -- comment out these lines to
    # see all the aisdc logging statements
    logging.getLogger("attack-reps").setLevel(logging.WARNING)
    logging.getLogger("prep-attack-data").setLevel(logging.WARNING)
    logging.getLogger("attack-from-preds").setLevel(logging.WARNING)

    model_filename = os.path.join(directory, target_model)
    print("Reading target model from " + model_filename)
    with open(model_filename, "rb") as f:
        target_model = pickle.load(f)

    print("Reading training/testing data from ./" + directory)
    train_x = np.loadtxt(os.path.join(directory, x_train))
    train_y = np.loadtxt(os.path.join(directory, y_train))
    test_x = np.loadtxt(os.path.join(directory, x_test))
    test_y = np.loadtxt(os.path.join(directory, y_test))

    target = Target(model=target_model)
    # Wrap the training and test data into the Target object
    target.add_processed_data(train_x, train_y, test_x, test_y)

    # Run the attack
    wca = WorstCaseAttack(
        n_dummy_reps=10, output_dir=directory, report_name=attack_output_name
    )
    wca.attack(target)

    _ = wca.make_report()

    lira_config = {
        "training_data_filename": "train_data.csv",
        "test_data_filename": "test_data.csv",
        "training_preds_filename": "train_preds.csv",
        "test_preds_filename": "test_preds.csv",
        "target_model": ["sklearn.ensemble", "RandomForestClassifier"],
        "target_model_hyp": {"min_samples_split": 2, "min_samples_leaf": 1},
    }

    with open(os.path.join(directory, "lira_config.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(lira_config))

    lira_attack_obj = LIRAAttack(
        n_shadow_models=100,
        attack_config_json_file_name=os.path.join(directory, "lira_config.json"),
        output_dir=directory,
        report_name=attack_output_name,
    )

    lira_attack_obj.attack(target)
    _ = lira_attack_obj.make_report()

    target.save(os.path.join(directory, "target"))

    t = GenerateTextReport()
    t.process_attack_target_json(
        os.path.join(directory, attack_output_name) + ".json",
        target_filename=os.path.join(directory, target_filename),
    )

    t.export_to_file(
        output_filename=os.path.join(directory, outfile),
        move_files=True,
        model_filename=model_filename,
    )

    print("Results written to " + os.path.join(directory, outfile))


def run_user_story(config: dict):
    """Main method to parse arguments and then invoke report generation."""

    generate_report(
        config["training_artefacts_dir"],
        config["target_model"],
        config["x_train_path"],
        config["y_train_path"],
        config["x_test_path"],
        config["y_test_path"],
        config["attack_output_name"],
        config["target_results"],
        config["outfile"],
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
