"""Example demonstrating the attribute inference attacks."""

import json
import os
import sys

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from aisdc.attacks import attribute_attack
from aisdc.attacks.target import Target

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

    # [Researcher] Define the classifier
    model = RandomForestClassifier(bootstrap=False)

    # [Researcher] Train the classifier
    model.fit(x_train, y_train)
    acc_train = model.score(x_train, y_train)
    acc_test = model.score(x_test, y_test)
    print(f"Base model train accuracy: {acc_train}")
    print(f"Base model test accuracy: {acc_test}")

    # [TRE / Researcher] Wrap the model and data in a Target object
    target = Target(model=model)
    target.name = "nursery"
    target.add_processed_data(x_train, y_train, x_test, y_test)
    target.add_raw_data(x, y, x_train_orig, y_train_orig, x_test_orig, y_test_orig)
    for i in range(n_features):
        target.add_feature(nursery_data.feature_names[i], indices[i], "onehot")

    print(f"Dataset: {target.name}")
    print(f"Features: {target.features}")
    print(f"x_train shape = {np.shape(target.x_train)}")
    print(f"y_train shape = {np.shape(target.y_train)}")
    print(f"x_test shape = {np.shape(target.x_test)}")
    print(f"y_test shape = {np.shape(target.y_test)}")

    # [TRE] Create the attack object with attack parameters
    attack_obj = attribute_attack.AttributeAttack(n_cpu=2, output_dir="outputs_aia")

    # [TRE] Run the attack
    attack_obj.attack(target)

    # [TRE] Grab the output
    output = attack_obj.make_report()  # also makes .pdf and .json files
    output = output["attack_experiment_logger"]["attack_instance_logger"]["instance_0"]

    # [TRE] explore the metrics
    print(attribute_attack.report_categorical(output))
    print(attribute_attack.report_quantitative(output))

    print("Programmatic example finished")
    print("****************************")

    print()
    print()
    print("Command line example starting")
    print("*****************************")

    # [Researcher] Dump the training and test predictions to .csv files
    target.save(path="aia_target")

    # [TRE] Runs the attack. This would be done on the command line, here we do that with os.system
    # [TRE] First they access the help to work out which parameters they need to set
    os.system(
        f"{sys.executable} -m aisdc.attacks.attribute_attack run-attack-from-configfile --help"
    )

    # [TRE] Then they run the attack

    # Example 1 to demonstrate running attack from configuration and target files
    config = {
        "n_cpu": 2,
        "output_dir": "outputs_aia",
    }

    with open("config_aia_cmd.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(config))

    os.system(
        f"{sys.executable} -m aisdc.attacks.attribute_attack run-attack-from-configfile "
        "--attack-config-json-file-name config_aia_cmd.json "
        "--attack-target-folder-path aia_target "
    )
