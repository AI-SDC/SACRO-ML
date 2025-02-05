"""RESEARCHER EXAMPLE FOR USER STORY 8.

This file is an example of a researcher creating/training a machine learning
model and to be released form a secure environment.

This specific example uses the nursery dataset: data is read in and
pre-processed, and a classifier is trained and tested on this dataset.

This example follows User Story 8.

Steps:

- Researcher creates a function to read and process a dataset, which a TRE can
  also use and call.
- Researcher creates and trains a classifier on this data.
- Researcher emails (or otherwise contacts) TRE to request the model be released.
- TREs will use this code/functions to test the model themselves.
"""

import logging
import os
import pickle

import pandas as pd
from data_processing_researcher import process_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def run_user_story():
    """Create and train a model to be released."""
    # This section is not necessary but helpful - cleans up files that are
    # created by sacroml
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

    # Write a function to pre-process the data that the TRE can call
    # (see data_processing_researcher.py)
    # Use the output of this function to split the data into training/testing sets
    # NOTE: to use this user story/script, the process_dataset function MUST:
    # take a single parameter (the data to be processed)
    # return a dictionary
    # which contains the keys
    # >>>  ['n_features_raw_data', 'X_transformed', 'y_transformed', 'train_indices']
    # as in this example

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

    # Create, train and test a model - replace this with your training and testing code
    hyperparameters = {}
    hyperparameters["min_samples_split"] = 5
    hyperparameters["min_samples_leaf"] = 5
    hyperparameters["max_depth"] = None
    hyperparameters["bootstrap"] = False

    # Build a model
    target_model = RandomForestClassifier(**hyperparameters)
    target_model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, target_model.predict(X_train))
    test_acc = accuracy_score(y_test, target_model.predict(X_test))
    print(f"Training accuracy on model: {train_acc:.2f}")
    print(f"Testing accuracy on model: {test_acc:.2f}")

    # Save your model somewhere a TRE can access
    filename = os.path.join(directory, "model.pkl")
    print("Saving model to " + filename)
    with open(filename, "wb") as file:
        pickle.dump(target_model, file)


if __name__ == "__main__":
    run_user_story()
