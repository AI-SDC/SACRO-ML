"""
Example demonstrating the attribute inference attacks.

Running
-------

Invoke this code from the root AI-SDC folder with
python -m examples.attribute_inference_example

"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from aisdc.attacks import attribute_attack, dataset  # pylint: disable = import-error

# pylint: disable = duplicate-code

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
    (x_train_orig, x_test_orig, y_train_orig, y_test_orig,) = train_test_split(
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

    # [TRE / Researcher] Wrap the data in a dataset object
    data = dataset.Data()
    data.name = "nursery"
    data.add_processed_data(x_train, y_train, x_test, y_test)
    data.add_raw_data(x, y, x_train_orig, y_train_orig, x_test_orig, y_test_orig)
    for i in range(n_features):
        data.add_feature(nursery_data.feature_names[i], indices[i], "onehot")

    print(f"Dataset: {data.name}")
    print(f"Features: {data.features}")
    print(f"x_train shape = {np.shape(data.x_train)}")
    print(f"y_train shape = {np.shape(data.y_train)}")
    print(f"x_test shape = {np.shape(data.x_test)}")
    print(f"y_test shape = {np.shape(data.y_test)}")

    # [Researcher] Define the classifier
    model = RandomForestClassifier(bootstrap=False)

    # [Researcher] Train the classifier
    model.fit(data.x_train, data.y_train)
    acc_train = model.score(data.x_train, data.y_train)
    acc_test = model.score(data.x_test, data.y_test)
    print(f"Base model train accuracy: {acc_train}")
    print(f"Base model test accuracy: {acc_test}")

    # [TRE] Define some attack parameters
    attack_args = attribute_attack.AttributeAttackArgs(
        n_cpu=7, report_name="aia_report"
    )

    # [TRE] Create the attack object
    attack_obj = attribute_attack.AttributeAttack(attack_args)

    # [TRE] Run the attack
    attack_obj.attack(data, model)

    # [TRE] Grab the output
    output = attack_obj.make_report()  # also makes .pdf and .json files
    output = output["attack_metrics"]

    # [TRE] explore the metrics
    print(attribute_attack.report_categorical(output))
    print(attribute_attack.report_quantitative(output))
