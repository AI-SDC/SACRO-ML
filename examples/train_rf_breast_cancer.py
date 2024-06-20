"""Train a Random Forest classifier on the breast cancer data."""

import logging

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from aisdc.attacks.target import Target

output_dir = "target_rf_breast_cancer"

if __name__ == "__main__":
    logging.info("Loading dataset")
    X, y = load_breast_cancer(return_X_y=True, as_frame=False)

    logging.info("Splitting data into training and test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    logging.info("Defining the model")
    model = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1)

    logging.info("Training the model")
    model.fit(X_train, y_train)

    logging.info("Wrapping the model and data in a Target object")
    target = Target(model=model)
    target.dataset_name = "breast cancer"
    target.add_processed_data(X_train, y_train, X_test, y_test)

    logging.info("Writing Target object to directory: '%s'", output_dir)
    target.save(output_dir)
