"""
User story 7 as researcher.

Details can be found here:
https://github.com/AI-SDC/AI-SDC/issues/141
"""
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from aisdc.attacks.target import Target  # pylint: disable=import-error
from aisdc.safemodel.classifiers import (
    SafeDecisionTreeClassifier,  # pylint: disable=import-error
)


def main():
    """Create and train model to be released."""
    directory = "training_artefacts/"
    print("Creating directory for training artefacts")

    if not os.path.exists(directory):
        os.makedirs(directory)

    print()
    print("Acting as researcher...")
    print()

    filename = "user_stories_resources/dataset_26_nursery.csv"
    print("Reading data from " + filename)
    data = pd.read_csv(filename)

    print()

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

    # Split into training and test sets
    # target model train / test split - these are strings
    (x_train_orig, x_test_orig, y_train_orig, y_test_orig) = train_test_split(
        x,
        y,
        test_size=0.5,
        stratify=y,
        shuffle=True,
    )

    # Preprocess dataset
    # one-hot encoding of features and integer encoding of labels
    label_enc = LabelEncoder()
    feature_enc = OneHotEncoder()
    x_train = feature_enc.fit_transform(x_train_orig).toarray()
    y_train = label_enc.fit_transform(y_train_orig)
    x_test = feature_enc.transform(x_test_orig).toarray()
    y_test = label_enc.transform(y_test_orig)

    logging.getLogger("attack-reps").setLevel(logging.WARNING)
    logging.getLogger("prep-attack-data").setLevel(logging.WARNING)
    logging.getLogger("attack-from-preds").setLevel(logging.WARNING)

    # Build a model
    model = SafeDecisionTreeClassifier(random_state=1)
    model.fit(x_train, y_train)
    model.request_release(path=directory, ext="pkl")

    # Wrap the model and data in a Target object
    target = Target(model=model)
    target.name = "nursery"
    target.add_processed_data(x_train, y_train, x_test, y_test)
    target.add_raw_data(x, y, x_train_orig, y_train_orig, x_test_orig, y_test_orig)
    for i in range(n_features):
        target.add_feature(data.columns[i], indices[i], "onehot")

    # NOTE: we assume here that the researcher does not use the target.save() function
    # and instead provides only the model

    logging.info("Dataset: %s", target.name)
    logging.info("Features: %s", target.features)
    logging.info("x_train shape = %s", np.shape(target.x_train))
    logging.info("y_train shape = %s", np.shape(target.y_train))
    logging.info("x_test shape = %s", np.shape(target.x_test))
    logging.info("y_test shape = %s", np.shape(target.y_test))


if __name__ == "__main__":
    main()
