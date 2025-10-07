"""Example saving predicted probabilities as csv files."""

import logging

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logging.info("Loading dataset")
    X, y = load_breast_cancer(return_X_y=True, as_frame=False)

    logging.info("Splitting data into training and test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    logging.info("Defining the model")
    model = RandomForestClassifier(bootstrap=False, min_samples_leaf=1)

    logging.info("Training the model")
    model.fit(X_train, y_train)

    logging.info("Saving predicted probabilities")
    proba_train = model.predict_proba(X_train)
    proba_test = model.predict_proba(X_test)

    np.savetxt("proba_train.csv", proba_train, delimiter=",")
    np.savetxt("proba_test.csv", proba_test, delimiter=",")
