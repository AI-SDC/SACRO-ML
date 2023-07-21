"""
User story 2 (best case) as TRE.

Details can be found here:
https://github.com/AI-SDC/AI-SDC/issues/141
"""

import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from aisdc.attacks.attack_report_formatter import GenerateTextReport
from aisdc.attacks.target import Target  # pylint: disable=import-error


def generate_report(
    directory,
    target_model,
    train_indices,
    test_indices,
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
    print()

    filename = directory + target_model
    print("Reading target model from " + filename)
    target_model = pickle.load(open(filename, "rb"))

    print("Reading training/testing indices from ./" + directory)
    indices_train = np.loadtxt(directory + train_indices)
    indices_test = np.loadtxt(directory + test_indices)

    filename = "user_stories_resources/dataset_26_nursery.csv"
    print("Reading data from " + filename)
    data = pd.read_csv(filename)

    print()

    print(data.head())
    print(indices_test[:10])
    print(indices_train[:10])

    y = np.asarray(data["class"])
    x = np.asarray(data.drop(columns=["class"], inplace=False))

    n_features = np.shape(x)[1]
    indices: list[list[int]] = [
        [0, 1, 2],  # parents
        [3, 4, 5, 6, 7],  # has_nurs
        [8, 9, 10, 11],  # form
        [12, 13, 14, 15],  # children
        [16, 17, 18],  # housing
        [19, 20],  # finance
        [21, 22, 23],  # social
        [24, 25, 26],  # health
    ]

    x_train_orig = np.asarray([x[int(i)] for i in indices_train])
    y_train_orig = np.asarray([y[int(i)] for i in indices_train])
    x_test_orig = np.asarray([x[int(i)] for i in indices_test])
    y_test_orig = np.asarray([y[int(i)] for i in indices_test])

    # Preprocess dataset
    # one-hot encoding of features and integer encoding of labels
    label_enc = LabelEncoder()
    feature_enc = OneHotEncoder()
    x_train = feature_enc.fit_transform(x_train_orig).toarray()
    y_train = label_enc.fit_transform(y_train_orig)
    x_test = feature_enc.transform(x_test_orig).toarray()
    y_test = label_enc.transform(y_test_orig)

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
        directory + attack_results, target_filename=directory + target_filename
    )

    t.export_to_file(output_filename=directory + outfile, move_files=True)

    print("Results written to " + directory + outfile)


def main():
    """Main method to parse arguments and then invoke report generation."""
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
        default="/model.pkl",
        help=("Filename of target model. Default = %(default)s."),
    )

    parser.add_argument(
        "--train_indices",
        type=str,
        action="store",
        dest="train_indices",
        required=False,
        default="indices_train.txt",
        help=("Filename for the saved training indices. Default = %(default)s."),
    )

    parser.add_argument(
        "--test_indices",
        type=str,
        action="store",
        dest="test_indices",
        required=False,
        default="indices_test.txt",
        help=("Filename for the saved testing indices. Default = %(default)s."),
    )

    parser.add_argument(
        "--attack_results",
        type=str,
        action="store",
        dest="attack_results",
        required=False,
        default="attack_results.json",
        help=("Filename for the saved JSON attack output. Default = %(default)s."),
    )

    parser.add_argument(
        "--target_results",
        type=str,
        action="store",
        dest="target_results",
        required=False,
        default="target.json",
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
            args.train_indices,
            args.test_indices,
            args.attack_results,
            args.target_results,
            args.outfile,
        )
    except AttributeError as e:  # pragma:no cover
        print("Invalid command. Try --help to get more details" f"error mesge is {e}")


if __name__ == "__main__":  # pragma:no cover
    main()
