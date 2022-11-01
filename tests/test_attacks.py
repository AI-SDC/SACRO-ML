"""Jim Smith October 2022"""
import math

import numpy as np
import pytest

from attacks import attack, dataset, mia_extremecase
from safemodel.classifiers import SafeDecisionTreeClassifier


def test_superclass():
    """Test that the exceptions are raised if the superclass is called in error"""
    my_attack = attack.Attack()
    dataset_obj = dataset.Data()
    target_obj = SafeDecisionTreeClassifier()
    with pytest.raises(NotImplementedError):
        my_attack.attack(target_obj, dataset_obj)
    with pytest.raises(NotImplementedError):
        print(str(my_attack))  # .__str__()


def test_mia_extremecase():
    """test the extreme case mia in the file of the same name"""

    # create actual values
    y = np.zeros(50000)
    y[:25] = 1
    # exactly right and wrong predictions
    right = np.zeros(50000)
    right[:25] = 1
    wrong = 1 - right

    # right predictions - triggers override for very small logp
    _, _, _, pval = mia_extremecase.min_max_disc(y, right)
    assert pval == -115.13

    # wrong predictions - probaility very close to 1 so logp=0
    _, _, _, pval = mia_extremecase.min_max_disc(y, wrong)
    assert math.isclose(pval, 0.0)
