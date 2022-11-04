"""Examples for using the likelihood ratio attack code.

This code simulates a MIA attack providing the attacker with as much information as possible.
i.e. they have a subset of rows that they _know_ were used for training. And a subset that they
know were not. They also have query access to the target model.

The attack proceeds as described in this paper:
https://arxiv.org/pdf/2112.03570.pdf

The steps are as follows:

1. The researcher partitions their data into training and testing subsets
2. The researcher trains their model
3. The TRE runs the attacks
   *Programmatically*
     1. The TRE calls the attack code.
     2. The TRE computes and inspects attack metrics.
   *Command line
     3. The researcher writes out their training and testing data, as well as the predictions
        that their target model makes on this data.
     4. The TRE create a config file for the attack, specifying the file names for the files created
        in the previous two steps, as well as specifications for the shadow models.
     5. The attack is run with a command line command, creating a report.


Running
-------

Invoke this code from the root AI-SDC folder with
python -m examples.lira_attack_example

"""
# pylint: disable = duplicate-code

# import json
# import os

# import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from attacks.dataset import Data  # pylint: disable = import-error
from attacks.likelihood_attack import (  # pylint: disable = import-error
    LIRAAttack,
    LIRAAttackArgs,
)

# from sklearn.svm import SVC


def test_lira_attack():
    """tests the lira code two ways"""
    args = LIRAAttackArgs(n_shadow_models=50, report_name="lira_example_report")
    attack_obj = LIRAAttack(args)
    attack_obj.setup_example_data()
    attack_obj.attack_from_config()
    attack_obj.example()

    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
    dataset = Data()
    dataset.add_processed_data(train_X, train_y, test_X, test_y)

    # target_model = SVC(gamma=0.1, probability=True)
    target_model = RandomForestClassifier(
        n_estimators=100, min_samples_split=2, min_samples_leaf=1
    )
    target_model.fit(train_X, train_y)

    args2 = LIRAAttackArgs(n_shadow_models=50, report_name="lira_example2_report")
    attack_obj2 = LIRAAttack(args2)
    attack_obj2.attack(dataset, target_model)
    output2 = attack_obj2.make_report()  # also makes .pdf and .json files
    _ = output2["attack_metrics"][0]
