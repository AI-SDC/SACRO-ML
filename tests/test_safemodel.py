import numpy as np
from sklearn import datasets

from safemodel.reporting import get_reporting_string
from safemodel.safemodel import SafeModel

class DummyClassifier( ):
    """Dummy Classifer that always returns predictions of zero"""
    def __init__(self,at_least_5=5,
                       at_most_5=5,
                       exactly_5=5):
        self.at_least_5= at_least_5
        self.at_most_5 = at_most_5
        self.exactly_5 = exactly_5
        
    def fit(x:np.ndarray,y:np.ndarray):
        pass
    def predict(x:np.ndarray):
        return np.ones(x.shape[0])
        
def get_data():
    """Returns data for testing."""
    iris = datasets.load_iris()
    x = np.asarray(iris["data"], dtype=np.float64)
    y = np.asarray(iris["target"], dtype=np.float64)
    x = np.vstack([x, (7, 2.0, 4.5, 1)])
    y = np.append(y, 4)
    return x, y

class SafeDummyClassifier(SafeModel, DummyClassifier):
    """Privacy protected dummy classifier."""

    def __init__(self, **kwargs) -> None:
        """Creates model and applies constraints to params."""
        SafeModel.__init__(self)
        self.basemodel_paramnames = [
            "at_least_5",
            "at_most 5",
            "exactly_5",
        ]

        the_kwds = {}
        for key, val in kwargs.items():
            if key in self.basemodel_paramnames:
                the_kwds[key] = val
        DummyClassifier.__init__(self, **the_kwds)
        self.model_type: str = "DummyClassifier"
        self.ignore_items = ["model_save_file", "basemodel_paramnames", "ignore_items"]
        #create an item to test additional_checks()
        self.examine_seperately_items = ["newthing"]
        self.newthing = {"myStringKey":"aString","myIntKey":42}
        
    def set_params(self,**kwargs):
        for key,val in kwargs.items():
            self.key = val
    def get_params(self):
        return self.__dict__

def test_params_checks():
    """test parameter  checks"""
    model=SafeDummyClassifier()
    
    notok_start = get_reporting_string(name="warn_possible_disclosure_risk")
    ok_start    = get_reporting_string(name="within_recommended_ranges")
    
    #1.0 ok
    print("1.0")
    msg,disclosive = model.preliminary_check()
    assert disclosive==False
    assert msg ==ok_start

    #1.1not ok- too low
    print ("1.1")
    model.at_least_5= 4
    msg,disclosive = model.preliminary_check()
    assert disclosive==True
    correct_msg = notok_start + get_reporting_string(
                                   name="less_than_min_value",
                                   key="at_least_5",
                                   cur_val=model.at_least_5 ,val=5)
    print(
          f'Correct msg:{correct_msg}\n'
          f'Actual  msg:{msg}\n'
         )
    assert msg==correct_msg 

    #1.2 not ok too high
    print("1.2")
    model.at_least_5 = 5
    model.at_most_5 = 6
    msg,disclosive = model.preliminary_check()
    assert disclosive==True
    correct_msg= notok_start +get_reporting_string(
                                    name="greater_than_max_value",
                                    key="at_most_5",
                                    cur_val=model.at_most_5 ,val=5)
    print(
          f'Correct msg:{correct_msg}\n'
          f'Actual  msg:{msg}\n'
         )
    assert msg==correct_msg 

    #1.3 not ok - not equal
    print("1.3")
    model.at_most_5 = 5
    model.exactly_5 = 6
    msg,disclosive = model.preliminary_check()
    assert disclosive==True
    correct_msg = notok_start + get_reporting_string(
                                    name="different_than_fixed_value",
                                    key="exactly_5",
                                    cur_val=model.exactly_5 ,val=5)       
    print(
          f'Correct msg:{correct_msg}\n'
          f'Actual  msg:{msg}\n'
         )
    assert msg==correct_msg 

    #1.4 not ok - wrong type
    print("1.4")
    model.exactly_5 = "five"
    msg,disclosive = model.preliminary_check()
    assert disclosive==True
    correct_msg = notok_start + get_reporting_string(
                                    name="different_than_recommended_type",
                                    key="exactly_5",
                                    cur_val=model.exactly_5,val="int")
    correct_msg = correct_msg + get_reporting_string(
                                    name="different_than_fixed_value",
                                    key="exactly_5",
                                    cur_val=model.exactly_5 ,val=5)  

    print(
          f'Correct msg:{correct_msg}\n'
          f'Actual  msg:{msg}\n'
         )
    assert msg==correct_msg 
