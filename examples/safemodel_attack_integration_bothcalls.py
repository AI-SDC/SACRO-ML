"""Example showing how to integrate attacks into safemodel classes."""

import logging

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from aisdc.attacks.target import Target
from aisdc.safemodel.classifiers import SafeDecisionTreeClassifier

if __name__ == "__main__":
    # [Researcher] Access a dataset
    nursery_data = fetch_openml(data_id=26, as_frame=True)
    x = np.asarray(nursery_data.data, dtype=str)
    y = np.asarray(nursery_data.target, dtype=str)

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

    # [Researcher] Split into training and test sets
    # target model train / test split - these are strings
    (
        x_train_orig,
        x_test_orig,
        y_train_orig,
        y_test_orig,
    ) = train_test_split(
        x,
        y,
        test_size=0.5,
        stratify=y,
        shuffle=True,
    )

    # [Researcher] Preprocess dataset
    # one-hot encoding of features and integer encoding of labels
    label_enc = LabelEncoder()
    feature_enc = OneHotEncoder()
    x_train = feature_enc.fit_transform(x_train_orig).toarray()
    y_train = label_enc.fit_transform(y_train_orig)
    x_test = feature_enc.transform(x_test_orig).toarray()
    y_test = label_enc.transform(y_test_orig)

    # [Researcher] Build a model
    model = SafeDecisionTreeClassifier(random_state=1)
    model.fit(x_train, y_train)
    msg, disclosive = model.preliminary_check()

    # [TRE / Researcher] Wrap the model and data in a Target object
    target = Target(model=model)
    target.name = "nursery"
    target.add_processed_data(x_train, y_train, x_test, y_test)
    target.add_raw_data(x, y, x_train_orig, y_train_orig, x_test_orig, y_test_orig)
    for i in range(n_features):
        target.add_feature(nursery_data.feature_names[i], indices[i], "onehot")

    logging.info("Dataset: %s", target.name)
    logging.info("Features: %s", target.features)
    logging.info("x_train shape = %s", np.shape(target.x_train))
    logging.info("y_train shape = %s", np.shape(target.y_train))
    logging.info("x_test shape = %s", np.shape(target.x_test))
    logging.info("y_test shape = %s", np.shape(target.y_test))

    # [TRE / Researcher] Perform disclosure checks
    SAVE_DIR = "mytest"
    SAVE_FILENAME = "direct_results"

    # check direct method
    print("==========> first running attacks explicitly via run_attack()")
    for attack_name in ["worst_case", "attribute", "lira"]:
        print(f"===> running {attack_name} attack directly")
        metadata = model.run_attack(target, attack_name, SAVE_DIR, SAVE_FILENAME)
        logging.info("metadata is:")
        for key, val in metadata.items():
            if isinstance(val, dict):
                logging.info(" %s ", key)
                for key1, val2 in val.items():
                    logging.info("  %s : %s", key1, val2)
            else:
                logging.info(" %s : %s", key, val)

    # now via request_release()
    print("===> now running attacks implicitly via request_release()")
    model.request_release(path=SAVE_DIR, ext="pkl", target=target)

    print(f"Please see the files generated in: {SAVE_DIR}")
