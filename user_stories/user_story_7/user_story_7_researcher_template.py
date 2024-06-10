"""RESEARCHER EXAMPLE FOR USER STORY 7.

This file is an example of a researcher creating/training a machine learning model and to be
released form a secure environment

This specific example uses the nursery dataset: data is read in and pre-processed, and a classifier
is trained and tested on this dataset.

This example follows User Story 7

NOTE: this user story is an example of a model that cannot be released since the researcher has not
provided enough data

Steps:

- Researcher creates and pre-processes a dataset
- Researcher creates and trains a classifier on this data
- Reasercher saves the model manually (e.g. using pickle, not through request_release() or similar)
- Researcher does not save the training/testing data, and therefore the TRE cannot verify the model
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from aisdc.attacks.target import Target  # pylint: disable=import-error
from aisdc.safemodel.classifiers import (  # pylint: disable=import-error
    SafeDecisionTreeClassifier,
)


def run_user_story():  # pylint: disable=too-many-locals
    """Create and train model to be released."""
    # This section is not necessary but helpful - cleans up files that are created by aisdc
    directory = "training_artefacts"
    print("Creating directory for training artefacts")

    if not os.path.exists(directory):
        os.makedirs(directory)

    print()
    print("Acting as researcher...")
    print()

    # Read in and pre-process the dataset - replace this with your data reading/pre-processing code
    filename = os.path.join(".", "user_stories_resources", "dataset_26_nursery.csv")
    print("Reading data from " + filename)
    data_df = pd.read_csv(filename)

    labels = np.asarray(data_df["class"])
    data = np.asarray(data_df.drop(columns=["class"], inplace=False))

    n_features = np.shape(data)[1]
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

    # Split into training and test sets
    (x_train_orig, x_test_orig, y_train_orig, y_test_orig) = train_test_split(
        data,
        labels,
        test_size=0.5,
        stratify=labels,
        shuffle=True,
    )

    # Preprocess dataset
    label_enc = LabelEncoder()
    feature_enc = OneHotEncoder()
    x_train = feature_enc.fit_transform(x_train_orig).toarray()
    y_train = label_enc.fit_transform(y_train_orig)
    x_test = feature_enc.transform(x_test_orig).toarray()
    y_test = label_enc.transform(y_test_orig)

    logging.getLogger("attack-reps").setLevel(logging.WARNING)
    logging.getLogger("prep-attack-data").setLevel(logging.WARNING)
    logging.getLogger("attack-from-preds").setLevel(logging.WARNING)

    # Create, train and test a model - replace this with your training and testing code
    model = SafeDecisionTreeClassifier(random_state=1)
    model.fit(x_train, y_train)
    model.request_release(path=directory, ext="pkl")

    # Wrap the model and data in a Target object
    target = Target(model=model)
    target.name = "nursery"
    target.add_processed_data(x_train, y_train, x_test, y_test)
    target.add_raw_data(
        data, labels, x_train_orig, y_train_orig, x_test_orig, y_test_orig
    )
    for i in range(n_features):
        target.add_feature(data_df.columns[i], indices[i], "onehot")

    # NOTE: we assume here that the researcher does not use the target.save() function
    # and instead provides only the model, preventing this model from being checked by the TRE

    logging.info("Dataset: %s", target.name)
    logging.info("Features: %s", target.features)
    logging.info("x_train shape = %s", np.shape(target.x_train))
    logging.info("y_train shape = %s", np.shape(target.y_train))
    logging.info("x_test shape = %s", np.shape(target.x_test))
    logging.info("y_test shape = %s", np.shape(target.y_test))


if __name__ == "__main__":
    run_user_story()
