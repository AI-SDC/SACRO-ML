"""
User story 2 (best case) as TRE.

Details can be found here:
https://github.com/AI-SDC/AI-SDC/issues/141

Running
-------

Invoke this code from the root AI-SDC folder with
python -m example_notebooks.user_stories.user_story_2.user_story_2_tre
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import yaml
from data_processing_researcher import process_dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from aisdc.attacks.attack_report_formatter import (  # pylint: disable=import-error
    GenerateTextReport,
)
from aisdc.attacks.target import Target  # pylint: disable=import-error


def generate_report(
    dataset_filename,
    directory,
    target_model,
    train_indices,
    test_indices,
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

    filename = os.path.join(directory, target_model)
    print("Reading target model from " + filename)
    with open(filename, "rb") as f:
        target_model = pickle.load(f)

    print("Reading training/testing indices from " + directory)
    indices_train = np.loadtxt(os.path.join(directory, train_indices))
    indices_test = np.loadtxt(os.path.join(directory, test_indices))

    print("Reading data from " + dataset_filename)
    data = pd.read_csv(dataset_filename)

    returned = process_dataset(data)

    x = returned["x"]
    y = returned["y"]
    indices = returned["indices"]
    indices_train = returned["indices_train"]
    indices_test = returned["indices_test"]
    x_train_orig = returned["x_train_orig"]
    y_train_orig = returned["y_train_orig"]
    x_test_orig = returned["x_test_orig"]
    y_test_orig = returned["y_test_orig"]
    x_train = returned["x_train"]
    y_train = returned["y_train"]
    x_test = returned["x_test"]
    y_test = returned["y_test"]
    n_features = returned["n_features"]

    # Wrap the model and data in a Target object
    target = Target(model=target_model)
    target.add_processed_data(x_train, y_train, x_test, y_test)
    target.add_raw_data(x, y, x_train_orig, y_train_orig, x_test_orig, y_test_orig)
    for i in range(n_features):
        target.add_feature(data.columns[i], indices[i], "onehot")

    SAVE_PATH = directory

    # TRE calls request_release()
    print("===> now running attacks implicitly via request_release()")
    target_model.request_release(path=SAVE_PATH, ext="pkl", target=target)

    print(f"Please see the files generated in: {SAVE_PATH}")

    t = GenerateTextReport()
    t.process_attack_target_json(
        os.path.join(directory, attack_results),
        target_filename=os.path.join(directory, target_filename),
    )

    t.export_to_file(output_filename=os.path.join(directory, outfile), move_files=True)

    print("Results written to " + str(os.path.join(directory, outfile)))


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

    generate_report(
        config["dataset_filename"],
        config["training_artefacts_dir"],
        config["target_model"],
        config["train_indices"],
        config["test_indices"],
        config["attack_results"],
        config["target_results"],
        config["outfile"],
    )


if __name__ == "__main__":  # pragma:no cover
    main()
