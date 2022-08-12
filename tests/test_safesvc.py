'''
Test the various models we have defined
'''

import unittest
from sklearn.svm import SVC
from safemodel.classifiers import SafeSVC
from sklearn import datasets
import numpy as np

def get_data():
    """Returns data for testing."""
    cancer = datasets.load_breast_cancer()
    x = np.asarray(cancer.data, dtype=np.float64)
    y = np.asarray(cancer.target, dtype=np.float64)

    return x, y

class TestDPSVC(unittest.TestCase):
    '''
    Test the differentially private SVC
    '''

    def test_run(self):
        '''
        Test the model runs
        '''
        dpsvc = SafeSVC()
        svc = SVC(kernel="rbf", gamma="scale", C=1., probability=True)
        x, y = get_data()

        dpsvc.fit(x, y)
        svc.fit(x, y)

        test_features = x
        dp_predictions = dpsvc.predict(test_features)
        sv_predictions = svc.predict(test_features)

        # Check that the two models have equal shape
        self.assertTupleEqual(dp_predictions.shape, sv_predictions.shape)

        dp_predprob = dpsvc.predict_proba(test_features)
        sv_predprob = svc.predict_proba(test_features)

        # Check that the two models have equal shape
        self.assertTupleEqual(dp_predprob.shape, sv_predprob.shape)
