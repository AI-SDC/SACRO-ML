"""
User story 3 as TRE
"""

import argparse
import json
import logging
import os
import pickle

import numpy as np

from aisdc.attacks.attack_report_formatter import GenerateJSONModule, GenerateTextReport
from aisdc.attacks.likelihood_attack import LIRAAttack
from aisdc.attacks.target import Target  # pylint: disable=import-error
from aisdc.attacks.worst_case_attack import WorstCaseAttack


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
):
    """
    Generate report based on target model
    """

    print()
    print("Acting as TRE...")
    print()

    directory = "training_artefacts/"

    if not os.path.exists(directory):
        os.makedirs(directory)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)

    # Suppress messages from AI-SDC -- comment out these lines to see all the aisdc logging statements
    logging.getLogger("attack-reps").setLevel(logging.WARNING)
    logging.getLogger("prep-attack-data").setLevel(logging.WARNING)
    logging.getLogger("attack-from-preds").setLevel(logging.WARNING)

    model_filename = directory + target_model
    print("Reading target model from " + model_filename)
    target_model = pickle.load(open(model_filename, "rb"))

    print("Reading training/testing data from ./" + directory)
    trainX = np.loadtxt(directory + x_train)
    trainy = np.loadtxt(directory + y_train)
    testX = np.loadtxt(directory + x_test)
    testy = np.loadtxt(directory + y_test)

    g = GenerateJSONModule(directory + attack_output_name)
    target = Target(model=target_model)
    # Wrap the training and test data into the Data object
    target.add_processed_data(trainX, trainy, testX, testy)

    # Run the attack
    wca = WorstCaseAttack(
        n_dummy_reps=10, report_name=directory + "/disclosive_model_raw_output"
    )
    wca.attack(target)

    json_out = wca.make_report(g)

    lira_config = {
        "training_data_filename": "train_data.csv",
        "test_data_filename": "test_data.csv",
        "training_preds_filename": "train_preds.csv",
        "test_preds_filename": "test_preds.csv",
        "target_model": ["sklearn.ensemble", "RandomForestClassifier"],
        "target_model_hyp": {"min_samples_split": 2, "min_samples_leaf": 1},
    }

    with open(directory + "lira_config.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(lira_config))

    lira_attack_obj = LIRAAttack(
        n_shadow_models=100,
        attack_config_json_file_name=directory + "lira_config.json",
    )

    lira_attack_obj.attack(target)
    output = lira_attack_obj.make_report(g)

    target.save(directory + "/target/")

    t = GenerateTextReport()
    t.process_attack_target_json(
        directory + attack_output_name, target_filename=directory + target_filename
    )

    t.export_to_file(output_filename=directory+outfile, model_filename=model_filename)

    print("Results written to " + directory+outfile)


def main():
    """main method to parse arguments and then invoke report generation"""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a risk report after request_release() has been called by researcher"
        )
    )

    parser.add_argument(
        "--training_artefacts_directory",
        type=str,
        action="store",
        dest="training_artefacts_directory",
        required=False,
        default="training_artefacts/",
        help=(
            "Folder containing training artefacts produced by researcher. Default = %(default)s."
        ),
    )

    parser.add_argument(
        "--target_model",
        type=str,
        action="store",
        dest="target_model",
        required=False,
        default="/disclosive_random_forest.sav",
        help=("Filename of target model. Default = %(default)s."),
    )

    parser.add_argument(
        "--x_train_path",
        type=str,
        action="store",
        dest="x_train_path",
        required=False,
        default="trainX.txt",
        help=("Filename for the saved training data. Default = %(default)s."),
    )

    parser.add_argument(
        "--y_train_path",
        type=str,
        action="store",
        dest="y_train_path",
        required=False,
        default="trainy.txt",
        help=("Filename for the saved training labels. Default = %(default)s."),
    )

    parser.add_argument(
        "--x_test_path",
        type=str,
        action="store",
        dest="x_test_path",
        required=False,
        default="testX.txt",
        help=("Filename for the saved testiing data. Default = %(default)s."),
    )

    parser.add_argument(
        "--y_test_path",
        type=str,
        action="store",
        dest="y_test_path",
        required=False,
        default="testy.txt",
        help=("Filename for the saved testing labels. Default = %(default)s."),
    )

    parser.add_argument(
        "--attack_output_name",
        type=str,
        action="store",
        dest="attack_output_name",
        required=False,
        default="/attack_output.json",
        help=(
            "Filename for the attack JSON output to be written to. Default = %(default)s."
        ),
    )

    parser.add_argument(
        "--target_results",
        type=str,
        action="store",
        dest="target_results",
        required=False,
        default="/target/target.json",
        help=("Filename for the saved JSON model output. Default = %(default)s."),
    )

    parser.add_argument(
        "--outfile",
        type=str,
        action="store",
        dest="outfile",
        required=False,
        default="summary.txt",
        help=(
            "Filename for the final results to be written to. Default = %(default)s."
        ),
    )

    args = parser.parse_args()

    try:
        generate_report(
            args.training_artefacts_directory,
            args.target_model,
            args.x_train_path,
            args.y_train_path,
            args.x_test_path,
            args.y_test_path,
            args.attack_output_name,
            args.target_results,
            args.outfile,
        )
    except AttributeError as e:  # pragma:no cover
        print("Invalid command. Try --help to get more details" f"error mesge is {e}")


if __name__ == "__main__":  # pragma:no cover
    main()
