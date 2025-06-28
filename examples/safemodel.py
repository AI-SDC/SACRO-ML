"""Example showing how to integrate attacks into safemodel classes."""

import logging

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sacroml.attacks.target import Target
from sacroml.safemodel.classifiers import SafeDecisionTreeClassifier

output_dir = "outputs_safemodel"

if __name__ == "__main__":
    logging.info("Loading dataset")
    nursery_data = fetch_openml(data_id=26, as_frame=True)
    X = np.asarray(nursery_data.data, dtype=str)
    y = np.asarray(nursery_data.target, dtype=str)

    n_features = np.shape(X)[1]
    indices = [
        [0, 1, 2],  # parents
        [3, 4, 5, 6, 7],  # has_nurs
        [8, 9, 10, 11],  # form
        [12, 13, 14, 15],  # children
        [16, 17, 18],  # housing
        [19, 20],  # finance
        [21, 22, 23],  # social
        [24, 25, 26],  # health
    ]

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

    logging.info("Defining the (safe) model")
    model = SafeDecisionTreeClassifier(random_state=1)

    logging.info("Training the model")
    model.fit(X_train, y_train)
    acc_train = model.score(X_train, y_train)
    acc_test = model.score(X_test, y_test)
    logging.info("Base model train accuracy: %.4f", acc_train)
    logging.info("Base model test accuracy: %.4f", acc_test)

    logging.info("Performing a preliminary check")
    msg, disclosive = model.preliminary_check()

    logging.info("Wrapping the model and data in a Target object")
    target = Target(
        model=model,
        dataset_name="nursery",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_train_orig=X_train_orig,
        y_train_orig=y_train_orig,
        X_test_orig=X_test_orig,
        y_test_orig=y_test_orig,
    )
    for i in range(n_features):
        target.add_feature(nursery_data.feature_names[i], indices[i], "onehot")

    logging.info("Dataset: %s", target.dataset_name)
    logging.info("Features: %s", target.features)
    logging.info("X_train shape: %s", str(target.X_train.shape))
    logging.info("y_train shape: %s", str(target.y_train.shape))
    logging.info("X_test shape: %s", str(target.X_test.shape))
    logging.info("y_test shape: %s", str(target.y_test.shape))

    logging.info("Performing disclosure checks")
    model.request_release(path=output_dir, ext="pkl", target=target)

    logging.info("Please see the files generated in: %s", output_dir)
