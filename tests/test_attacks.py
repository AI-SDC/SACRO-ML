from attacks import attack,dataset
import pytest
from safemodel.classifiers import SafeDecisionTreeClassifier

#test that superclass cannot be called
def test_superclass():
    my_attack = attack.Attack()
    dataset_obj = dataset.Data()
    target_obj=SafeDecisionTreeClassifier()
    with pytest.raises(NotImplementedError) :
        my_attack.attack(dataset_obj,dataset_obj)
    with pytest.raises(NotImplementedError) :
        my_attack.__str__()
