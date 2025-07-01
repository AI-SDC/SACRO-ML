"""Train a Random Forest classifier on the nursery dataset."""

import logging

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sacroml.attacks.target import Target

output_dir = "target_rf_nursery"

if __name__ == "__main__":
    logging.info("Loading dataset")
    nursery_data = fetch_openml(data_id=26, as_frame=True)
    X = np.asarray(nursery_data.data, dtype=str)
    y = np.asarray(nursery_data.target, dtype=str)

    logging.info("Splitting data into training and test sets")
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=0.5, stratify=y, shuffle=True
    )

    logging.info("Preprocessing dataset")
    label_enc = LabelEncoder()
    feature_enc = OneHotEncoder()
    X_train = feature_enc.fit_transform(X_train_orig).toarray()
    y_train = label_enc.fit_transform(y_train_orig)
    X_test = feature_enc.transform(X_test_orig).toarray()
    y_test = label_enc.transform(y_test_orig)

    logging.info("Defining the model")
    model = RandomForestClassifier(bootstrap=False)

    logging.info("Training the model")
    model.fit(X_train, y_train)
    acc_train = model.score(X_train, y_train)
    acc_test = model.score(X_test, y_test)
    logging.info("Base model train accuracy: %.4f", acc_train)
    logging.info("Base model test accuracy: %.4f", acc_test)

    logging.info("Wrapping the model and data in a Target object")
    target = Target(
        model=model,
        dataset_name="nursery",
        # processed data
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        # original unprocessed data
        X_train_orig=X_train_orig,
        y_train_orig=y_train_orig,
        X_test_orig=X_test_orig,
        y_test_orig=y_test_orig,
    )

    logging.info("Wrapping feature details and encoding for attribute inference")
    feature_indices = [
        [0, 1, 2],  # parents
        [3, 4, 5, 6, 7],  # has_nurs
        [8, 9, 10, 11],  # form
        [12, 13, 14, 15],  # children
        [16, 17, 18],  # housing
        [19, 20],  # finance
        [21, 22, 23],  # social
        [24, 25, 26],  # health
    ]
    for i, index in enumerate(feature_indices):
        target.add_feature(
            name=nursery_data.feature_names[i],
            indices=index,
            encoding="onehot",
        )

    logging.info("Writing Target object to directory: '%s'", output_dir)
    target.save(output_dir)
