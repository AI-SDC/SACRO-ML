"""Jim Smith October 2022"""
import pytest
from attacks import attack,dataset

from safemodel.classifiers import SafeDecisionTreeClassifier

def test_superclass():
    """Test that the exceptions are raised if the superclass is called in error"""
    my_attack = attack.Attack()
    dataset_obj = dataset.Data()
    target_obj=SafeDecisionTreeClassifier()
    with pytest.raises(NotImplementedError) :
        my_attack.attack(target_obj,dataset_obj)
    with pytest.raises(NotImplementedError) :
        print(str(my_attack))#.__str__()
