"""
Run a set of experiments from json file.
This will create a combination of hyperparameters for each
classifier and generate a results table to summarise them.
"""

import argparse
import hashlib
import importlib
import json
import logging
import os
import sys
from itertools import product
from typing import TypedDict

import joblib
import pandas as pd
from experiments.metrics import get_metrics # pylint: disable=import-error
#from aisdc.attacks.target import Target  # pylint: disable = import-error
from aisdc.preprocessing import loaders  # pylint: disable = import-error
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logger = logging.getLogger(__file__)


class ResultsEntry:  # pylint: disable=too-few-public-methods, too-many-locals, too-many-branches
    """Class that experimental results are put into. Provides them back as a dataframe."""

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        model_data_param_id,
        param_id,
        dataset_name,
        scenario_name=None,
        classifier_name=None,
        target_clf_file=None,
        attack_classifier_name=None,
        attack_clf_file=None,
        repetition=None,
        params=None,
        target_metrics=None,
        target_generalisation_error=None,
        attack_metrics=None,
        mia_hyp=None,
    ):
        if params is None:
            params = {}
        if target_metrics is None:
            target_metrics = {}
        #else:
        #    target_metrics = {f"target_{k}": v for k, v in target_metrics.items()}
        if attack_metrics is None:
            attack_metrics = {}
        else:
            attack_metrics = {f"attack_{k}": v for k, v in attack_metrics.items()}
        if mia_hyp is None:
            mia_hyp = {}
        else:
            mia_hyp = {f"mia_hyp_{k}": v for k, v in mia_hyp.items()}

        self.metadata = {
            "dataset": dataset_name,
            "scenario": scenario_name,
            "target_classifier": classifier_name,
            "target_clf_file": target_clf_file,
            "attack_classifier": attack_classifier_name,
            "attack_clf_file": attack_clf_file,
            "repetition": repetition,
            "target_generalisation_error": target_generalisation_error,
            "model_data_param_id": model_data_param_id,
            "param_id": param_id,
        }
        self.params = params
        self.target_metrics = target_metrics
        self.attack_metrics = attack_metrics
        self.mia_hyp = mia_hyp

    def to_dataframe(self):
        """Convert entry into a dataframe with a single row."""
        return pd.DataFrame.from_dict(
            {
                **self.metadata,
                **self.params,
                **self.target_metrics,
                **self.attack_metrics,
                **self.mia_hyp,
            },
            orient="index",
        ).T


def create_directory(directory: str):
    """Create a new directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.mkdir(directory)


def read_experiment_config_file(experiment_config_file: str) -> TypedDict:
    """Load the experiments configuration."""
    with open(experiment_config_file, encoding="utf-8") as config_handle:
        return json.loads(config_handle.read())


def run_loop(  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    experiment_config_file: str,
) -> pd.DataFrame:
    """
    Run the experimental loop defined in the json config_file. Return
    a dataframe of results (which is also saved as a file).
    """
    logger.info("Running experiments with config: %s", experiment_config_file)

    config = read_experiment_config_file(experiment_config_file)

    # Set path to store/load results
    results_filename = config["results_filename"]
    experiments_path = str(config["path"])
    logger.info(
        "Creating target model folder if it doesn't exist: %s", experiments_path
    )
    create_directory(experiments_path)
    target_model_path = os.path.join(experiments_path, "target_models")

    # Get classifiers
    classifier_strings = config["classifiers"]
    classifiers = {}
    for module_name, class_name in classifier_strings:
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        classifiers[class_name] = class_

    # Get combination of experimental parameters
    #   for each classifier
    experiment_params = config["experiment_params"]

    # Get seed to reproduce data split
    if "reproducible_split" in config:
        seed = config["reproducible_split"]
    else:
        seed = 42

    scenarios = config["scenarios"]
    datasets = config["datasets"]
    results_filename = config["results_filename"]
    # Set proportion of data for test
    test_prop = 0.3

    results = (
        pd.DataFrame()
    )  # create an empty data frame to store the experimental results

    for dataset in datasets:  # pylint: disable=too-many-nested-blocks
        print(dataset)
        # load data

        ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))
        data_folder = os.path.join(ROOT_FOLDER, "experiments", "data")
        features, labels = loaders.get_data_sklearn(dataset, data_folder=data_folder)
        features = features.values

        # Reproduce data split
        train_X, test_X, train_y, test_y = train_test_split(
            features,
            labels.values.ravel(),
            test_size=test_prop,
            stratify=labels,
            random_state=seed,
            shuffle=True,
        )

        # Hyper-parameter combinations
        for classifier_name, clf_class in classifiers.items():
            logger.info("Classifier: %s", classifier_name)
            all_combinations = product(*experiment_params[classifier_name].values())
            for _, combination in enumerate(all_combinations):
                logger.info("combination: %s %s", _, combination)
                # Turn this particular combination into a dictionary
                params = dict(
                    zip(experiment_params[classifier_name].keys(), combination)
                )
                # Unique ID for target model file
                hashstr = f"{str(params)}"
                param_id = hashlib.sha256(hashstr.encode("utf-8")).hexdigest()
                hashstr = f"{dataset} {classifier_name} {str(params)} {seed}"
                target_model_id = hashlib.sha256(hashstr.encode("utf-8")).hexdigest()
                target_model_filename = os.path.join(
                    target_model_path, f"{target_model_id}.pkl"
                )  # pylint: disable = line-too-long
                create_directory(target_model_path)

                # LOAD or CREATE target model
                if os.path.isfile(target_model_filename):
                    # load the target model file
                    target_model = joblib.load(target_model_filename)
                else:
                    target_model = clf_class(**params)
                    # Train the target model
                    target_model.fit(train_X, train_y)
                    # save the target model
                    # joblib.dump(target_model, target_model_filename)

                # Compute the predictions on the training and test sets
                # train_preds = target_model.predict_proba(train_X)
                # test_preds = target_model.predict_proba(test_X)

                # Wrap the model and data in a Target object
                #target = Target(model=target_model)
                #target.add_processed_data(train_X, train_y, test_X, test_y)

                # Get target metrics
                target_metrics = {f"target_{key}": val for key, val in \
                            get_metrics(target_model, test_X, test_y).items()}
                target_train_metrics = {f"target_train_{key}": val for key, val in \
                            get_metrics(target_model, train_X, train_y).items()}
                target_metrics = {**target_metrics, **target_train_metrics}           

                attack_results = ResultsEntry(  # f'full_id_{i}',
                                model_data_param_id=target_model_id,
                                param_id=param_id,
                                dataset_name=dataset,
                                classifier_name=classifier_name,
                                target_clf_file=target_model_filename,
                                params=params,
                                target_metrics=target_metrics,
                            )
                results = pd.concat(
                                [results, attack_results.to_dataframe()],
                                ignore_index=True,
                                sort=False,
                            )
                        

    results.to_csv(results_filename, index=False)
    return results


def main():
    """Invoke the loop."""
    parser = argparse.ArgumentParser(
        description=("Run predictions with the parameters defined in the config file.")
    )
    parser.add_argument(
        action="store",
        dest="config_filename",
        help=(
            "json formatted file that contain hyper-parameter for loop search. "
            'It is assumed the file is located in "experiments" directory, so please provide path '
            "and filename, e.g. RF/randomForest_config.json"
        ),
    )

    args = parser.parse_args()
    config_file = args.config_filename

    run_loop(config_file)


if __name__ == "__main__":
    main()
