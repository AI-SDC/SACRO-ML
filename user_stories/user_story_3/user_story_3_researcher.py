"""
RESEARCHER EXAMPLE FOR USER STORY 3.

This file is an example of a researcher creating/training a machine learning model and to be
released form a secure environment

This specific example uses the nursery dataset: data is read in and pre-processed, and a classifier
is trained and tested on this dataset.

This example follows User Story 3

Steps:

- Researcher creates and pre-processes a dataset
- Researcher creates and trains a classifier on this data
- Reasercher saves the model manually (e.g. using pickle, not through request_release() or similar)
- Researcher emails (or otherwise contacts) TRE to request the model be released
- TREs will use this model and data to test the model themselves
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def run_user_story():  # pylint: disable=too-many-locals
    """Create and train a model to be released."""

    # This section is not necessary but helpful - cleans up files that are created by aisdc
    directory = "training_artefacts"
    print("Creating directory for training artefacts")

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Read in and pre-process the dataset - replace this with your data reading/pre-processing code
    filename = os.path.join(".", "user_stories_resources", "dataset_26_nursery.csv")
    print("Reading data from " + filename)
    data = pd.read_csv(filename)

    target_encoder = LabelEncoder()
    target_vals = target_encoder.fit_transform(data["class"].values)
    target_dataframe = pd.DataFrame({"class": target_vals})
    data = data.drop(columns=["class"], inplace=False)

    feature_encoder = OneHotEncoder()
    x_encoded = feature_encoder.fit_transform(data).toarray()
    feature_dataframe = pd.DataFrame(
        x_encoded, columns=feature_encoder.get_feature_names_out()
    )

    x_train, x_test, y_train, y_test = train_test_split(
        feature_dataframe.values,
        target_dataframe.values.flatten(),
        test_size=0.7,
        random_state=42,
    )

    # Save the training and test data to a file which a TRE can access
    print("Saving training/testing data to ./" + directory)
    np.savetxt(os.path.join(directory, "x_train.txt"), x_train, fmt="%d")
    np.savetxt(os.path.join(directory, "y_train.txt"), y_train, fmt="%d")
    np.savetxt(os.path.join(directory, "x_test.txt"), x_test, fmt="%d")
    np.savetxt(os.path.join(directory, "y_test.txt"), y_test, fmt="%d")

    # Create, train and test a model - replace this with your training and testing code
    hyperparameters = {}
    hyperparameters["min_samples_split"] = 5
    hyperparameters["min_samples_leaf"] = 5
    hyperparameters["max_depth"] = None
    hyperparameters["bootstrap"] = False

    target_model = RandomForestClassifier(**hyperparameters)
    target_model.fit(x_train, y_train)

    train_acc = accuracy_score(y_train, target_model.predict(x_train))
    test_acc = accuracy_score(y_test, target_model.predict(x_test))
    print(f"Training accuracy on model: {train_acc:.2f}")
    print(f"Testing accuracy on model: {test_acc:.2f}")

    # Save your model somewhere a TRE can access
    filename = os.path.join(directory, "model.pkl")
    print("Saving model to " + filename)
    with open(filename, "wb") as file:
        pickle.dump(target_model, file)


if __name__ == "__main__":
    run_user_story()
