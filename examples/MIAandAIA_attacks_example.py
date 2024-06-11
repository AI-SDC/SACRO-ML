"""Examples for running multiple attacks.

Includes the Membership Inference Attack and the Attribute Inference Attack
with a single configuration file have multiple configurations.

Below, [Researcher] and [TRE] are used to denote which task is performed by whom.
"""

import json
import os
import sys

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from aisdc.attacks.multiple_attacks import (  # pylint: disable = import-error
    ConfigFile,
    MultipleAttacks,
)
from aisdc.attacks.target import Target  # pylint: disable = import-error

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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

    # [TRE / Researcher] Wrap the model and data in a Target object
    target = Target(model=model)
    target.name = "nursery"
    target.add_processed_data(x_train, y_train, x_test, y_test)
    target.add_raw_data(x, y, x_train_orig, y_train_orig, x_test_orig, y_test_orig)
    for i in range(n_features):
        target.add_feature(nursery_data.feature_names[i], indices[i], "onehot")

    # [Researcher] Dump the target model and target data
    target.save(path="target")

    # [TRE / Researcher] Wrap the model and data in a Target object
    # Instantiating a ConfigFile instance to add configurations
    # (i.e., configuration dictionaries or a configuration file)
    # to a single configuration file and then running attacks
    configfile_obj = ConfigFile(
        filename="single_config.json",
    )

    # Adding three worst-case attack configuration dictionaries to the JSON file
    config = {
        "n_reps": 10,
        "n_dummy_reps": 1,
        "p_thresh": 0.05,
        "test_prop": 0.5,
        "train_beta": 5,
        "test_beta": 2,
        "output_dir": "outputs_multiple_attacks",
        "report_name": "report_multiple_attacks",
    }
    configfile_obj.add_config(config, "worst_case")

    config = {
        "n_reps": 20,
        "n_dummy_reps": 1,
        "p_thresh": 0.05,
        "test_prop": 0.5,
        "train_beta": 5,
        "test_beta": 2,
        "output_dir": "outputs_multiple_attacks",
        "report_name": "report_multiple_attacks",
    }
    configfile_obj.add_config(config, "worst_case")

    config = {
        "n_reps": 10,
        "n_dummy_reps": 1,
        "p_thresh": 0.05,
        "test_prop": 0.5,
        "train_beta": 5,
        "test_beta": 2,
        "output_dir": "outputs_multiple_attacks",
        "report_name": "report_multiple_attacks",
        "training_preds_filename": "train_preds.csv",
        "test_preds_filename": "test_preds.csv",
        "attack_metric_success_name": "P_HIGHER_AUC",
        "attack_metric_success_thresh": 0.05,
        "attack_metric_success_comp_type": "lte",
        "attack_metric_success_count_thresh": 2,
        "attack_fail_fast": True,
    }
    configfile_obj.add_config(config, "worst_case")

    # Adding two lira attack configuration dictionaries to the JSON file
    config = {
        "n_shadow_models": 100,
        "output_dir": "outputs_multiple_attacks",
        "report_name": "report_multiple_attacks",
        "training_data_filename": "train_data.csv",
        "test_data_filename": "test_data.csv",
        "training_preds_filename": "train_preds.csv",
        "test_preds_filename": "test_preds.csv",
        "target_model": ["sklearn.ensemble", "RandomForestClassifier"],
        "target_model_hyp": {"min_samples_split": 2, "min_samples_leaf": 1},
    }
    configfile_obj.add_config(config, "lira")

    config = {
        "n_shadow_models": 150,
        "output_dir": "outputs_multiple_attacks",
        "report_name": "report_multiple_attacks",
        "shadow_models_fail_fast": True,
        "n_shadow_rows_confidences_min": 10,
        "training_data_filename": "train_data.csv",
        "test_data_filename": "test_data.csv",
        "training_preds_filename": "train_preds.csv",
        "test_preds_filename": "test_preds.csv",
        "target_model": ["sklearn.ensemble", "RandomForestClassifier"],
        "target_model_hyp": {"min_samples_split": 2, "min_samples_leaf": 1},
    }
    configfile_obj.add_config(config, "lira")

    # Adding a lira JSON configuration file to a configuration file
    # having multiple attack configurations
    config = {
        "n_shadow_models": 120,
        "output_dir": "outputs_multiple_attacks",
        "report_name": "report_multiple_attacks",
        "shadow_models_fail_fast": True,
        "n_shadow_rows_confidences_min": 10,
        "training_data_filename": "train_data.csv",
        "test_data_filename": "test_data.csv",
        "training_preds_filename": "train_preds.csv",
        "test_preds_filename": "test_preds.csv",
        "target_model": ["sklearn.ensemble", "RandomForestClassifier"],
        "target_model_hyp": {"min_samples_split": 2, "min_samples_leaf": 1},
    }
    with open("lira_config.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(config))
    configfile_obj.add_config("lira_config.json", "lira")

    # Adding an attribute inference attack configuration dictionary to the JSON file
    config = {
        "n_cpu": 2,
        "output_dir": "outputs_multiple_attacks",
        "report_name": "report_multiple_attacks",
    }
    configfile_obj.add_config(config, "attribute")

    # Instantiating MultipleAttacks object specifying a single configuration file
    # (with multiple attacks configurations) and a single JSON output file
    attack_obj = MultipleAttacks(config_filename="single_config.json")
    attack_obj.attack(target)

    # [TRE] Runs the attack. This would be done on the command line, here we do that with os.system
    # [TRE] First they access the help to work out which parameters they need to set
    os.system(
        f"{sys.executable} -m aisdc.attacks.multiple_attacks run-attack-from-configfile --help"
    )

    # # [TRE] Then they run the attack
    os.system(
        f"{sys.executable} -m aisdc.attacks.multiple_attacks run-attack-from-configfile "
        "--attack-config-json-file-name single_config.json "
        "--attack-target-folder-path target "
    )
