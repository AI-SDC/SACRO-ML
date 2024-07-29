"""RESEARCHER EXAMPLE FOR USER STORY 2.

This file is an example of a researcher creating/training a machine learning
model and to be released form a secure environment.

This specific example uses the nursery dataset: data is read in and
pre-processed, and a classifier is trained and tested on this dataset.

This example follows User Story 2.

Steps:

- Researcher creates a function to read and process a dataset, which a TRE can
  also use and call.
- Researcher creates and trains a classifier on this data.
- Researcher emails (or otherwise contacts) TRE to request the model be released.
- TREs will use this code/functions to test the model themselves.
"""

import logging
import os

import numpy as np
import pandas as pd
from aisdc.attacks.target import Target
from aisdc.safemodel.classifiers import SafeDecisionTreeClassifier
from data_processing_researcher import process_dataset


def run_user_story():
    """Create and train a model to be released."""
    # This section is not necessary but helpful - cleans up files that are
    # created by aisdc
    directory = "training_artefacts"
    print("Creating directory for training artefacts")

    if not os.path.exists(directory):
        os.makedirs(directory)

    print()
    print("Acting as researcher...")
    print()

    # Read in and pre-process the dataset - replace this with your dataset
    filename = os.path.join(".", "user_stories_resources", "dataset_26_nursery.csv")
    print("Reading data from " + filename)
    data = pd.read_csv(filename)

    # Write a function to pre-process the data that the TRE can call (see
    # data_processing_researcher.py) Use the output of this function to split
    # the data into training/testing sets.

    # NOTE: to use this user story/script, the process_dataset function MUST:
    # take a single parameter (the data to be processed) return a dictionary
    # which contains the keys:
    # >>> ['n_features_raw_data', 'X_transformed', 'y_transformed', 'train_indices']
    # as in this example.

    returned = process_dataset(data)

    X_transformed = returned["X_transformed"]
    y_transformed = returned["y_transformed"]

    train_indices = set(returned["train_indices"])

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for i, label in enumerate(y_transformed):
        if i in train_indices:
            X_train.append(X_transformed[i])
            y_train.append(label)
        else:
            X_test.append(X_transformed[i])
            y_test.append(label)

    logging.getLogger("attack-reps").setLevel(logging.WARNING)
    logging.getLogger("prep-attack-data").setLevel(logging.WARNING)
    logging.getLogger("attack-from-preds").setLevel(logging.WARNING)

    # Build a model and request its release
    model = SafeDecisionTreeClassifier(random_state=1)
    model.fit(X_train, y_train)
    model.request_release(path=directory, ext="pkl")

    # Wrap the model and data in a Target object
    target = Target(model=model)
    target.name = "nursery"
    target.add_processed_data(X_train, y_train, X_test, y_test)

    # NOTE: we assume here that the researcher does not use the target.save()
    # function and instead provides only the model and the list of indices
    # which have been used to split the dataset, which will allow a TRE to
    # re-create the input data used in training.

    logging.info("Dataset: %s", target.name)
    logging.info("Features: %s", target.features)
    logging.info("X_train shape = %s", np.shape(target.X_train))
    logging.info("y_train shape = %s", np.shape(target.y_train))
    logging.info("X_test shape = %s", np.shape(target.X_test))
    logging.info("y_test shape = %s", np.shape(target.y_test))


if __name__ == "__main__":
    run_user_story()
