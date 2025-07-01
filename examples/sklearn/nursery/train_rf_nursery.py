"""Train a Random Forest classifier on the nursery dataset."""

import logging

from dataset import Nursery
from sklearn.ensemble import RandomForestClassifier

from sacroml.attacks.target import Target

output_dir = "target_rf_nursery"

if __name__ == "__main__":
    logging.info("Loading dataset")
    handler = Nursery()

    logging.info("Splitting data into training and test sets")
    indices_train, indices_test = handler.get_train_test_indices()

    logging.info("Getting data")
    X, y = handler.get_data()
    X_train, y_train = handler.get_subset(X, y, indices_train)
    X_test, y_test = handler.get_subset(X, y, indices_test)

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
        dataset_name="Nursery",  # Must match the class name in dataset module
        dataset_module_path="dataset.py",
        indices_train=indices_train,
        indices_test=indices_test,
    )

    logging.info("Wrapping feature details and encoding for attribute inference")
    for i, index in enumerate(handler.feature_indices):
        target.add_feature(
            name=handler.feature_names[i],
            indices=index,
            encoding="onehot",
        )

    logging.info("Writing Target object to directory: '%s'", output_dir)
    target.save(output_dir)
