"""RESEARCHER EXAMPLE FOR USER STORY 1.

This file is an example of a researcher creating/training a machine learning
model and requesting for it to be released.

This specific example uses the nursery dataset: data is read in and
pre-processed, and a classifier is trained and tested on this dataset.

This example follows User Story 1.

Steps:

- Researcher reads in data and processes it.
- Researcher creates and trains a classifier.
- Researcher runs experiments themselves to check if their model is disclosive
  or not.
- Once satisfied, researcher calls request_release() to make it ready for TRE
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sacroml.attacks.target import Target
from sacroml.safemodel.classifiers import SafeDecisionTreeClassifier


def main():
    """Create and train a model to be released."""
    # This section is not necessary but helpful - cleans up files that are
    # created by sacroml
    save_directory = "training_artefacts"
    print("Creating directory for training artefacts")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    print()
    print("Acting as researcher...")
    print()

    # Read in and pre-process the dataset - replace this with your data
    # reading/pre-processing code
    print(os.getcwd())
    filename = os.path.join(".", "user_stories_resources", "dataset_26_nursery.csv")
    print("Reading data from " + filename)
    data_df = pd.read_csv(filename)

    print()

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

    (
        X_train_orig,
        X_test_orig,
        y_train_orig,
        y_test_orig,
    ) = train_test_split(
        data,
        labels,
        test_size=0.5,
        stratify=labels,
        shuffle=True,
    )

    label_enc = LabelEncoder()
    feature_enc = OneHotEncoder()
    X_train = feature_enc.fit_transform(X_train_orig).toarray()
    y_train = label_enc.fit_transform(y_train_orig)
    X_test = feature_enc.transform(X_test_orig).toarray()
    y_test = label_enc.transform(y_test_orig)

    logging.getLogger("attack-reps").setLevel(logging.WARNING)
    logging.getLogger("prep-attack-data").setLevel(logging.WARNING)
    logging.getLogger("attack-from-preds").setLevel(logging.WARNING)

    # Create and train a SafeDecisionTree classifier on the above data
    model = SafeDecisionTreeClassifier(random_state=1)
    model.fit(X_train, y_train)

    # Run a preliminary check to make sure the model is not disclosive
    _, _ = model.preliminary_check()

    # Wrap the model and data in a Target object
    # needed in order to call request_release()
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
    for i in range(n_features):
        target.add_feature(data_df.columns[i], indices[i], "onehot")

    logging.info("Dataset: %s", target.dataset_name)
    logging.info("Features: %s", target.features)
    logging.info("X_train shape: %s", str(target.X_train.shape))
    logging.info("y_train shape: %s", str(target.y_train.shape))
    logging.info("X_test shape: %s", str(target.X_test.shape))
    logging.info("y_test shape: %s", str(target.y_test.shape))

    # Researcher can check for themselves whether their model passes individual
    # disclosure checks.Leave this code as-is for output disclosure checking.
    print("==========> first running attacks explicitly via run_attack()")
    for attack_name in ["worst_case", "attribute", "lira"]:
        print(f"===> running {attack_name} attack directly")
        metadata = model.run_attack(target, attack_name, save_directory)
        logging.info("metadata is:")
        for key, val in metadata.items():
            if isinstance(val, dict):
                logging.info(" %s ", key)
                for key1, val2 in val.items():
                    logging.info("  %s : %s", key1, val2)
            else:
                logging.info(" %s : %s", key, val)

    # Modify/re-run all of the above code until you're happy with the model
    # you've created. If the tests do not pass, try changing the model or
    # hyperparameters until the tests pass. When you are satisfied and ready to
    # release your model, call the request release() function with the Target
    # class you created above.

    # This code will run checks for the TRE staff.

    # NOTE: you should only do this when you have confirmed that the above
    # tests pass. You would not normally waste your and TRE time calling this
    # unless you have already checked that your model is not disclosive or can
    # provide a justification for an exception request.

    print("===> now running attacks implicitly via request_release()")
    model.request_release(path=save_directory, ext="pkl", target=target)

    # The files generated can be found in this file location.
    print(f"Please see the files generated in: {save_directory}")


if __name__ == "__main__":
    main()
