"""attack.py - base class for an attack object"""

import sklearn

from aisdc.attacks.dataset import Data


class Attack:
    """Base (abstract) class to represent an attack"""

    def attack(self, dataset: Data, target_model: sklearn.base.BaseEstimator) -> None:
        """Method to run an attack"""
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError
