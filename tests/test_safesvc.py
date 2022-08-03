'''
Test the various models we have defined
'''

import unittest
from sklearn.svm import SVC
from safemodel.classifiers import SafeSVC
from preprocessing.loaders import get_data_sklearn

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
        mimic_features, mimic_labels = get_data_sklearn("minmax mimic2-iaccd")

        dpsvc.fit(mimic_features.values[:100, :], mimic_labels.values[:100, :].flatten())
        svc.fit(mimic_features.values[:100, :], mimic_labels.values[:100, :].flatten())

        test_features = mimic_features.values[101:201, :]
        dp_predictions = dpsvc.predict(test_features)
        sv_predictions = svc.predict(test_features)

        # Check that the two models have equal shape
        self.assertTupleEqual(dp_predictions.shape, sv_predictions.shape)

        dp_predprob = dpsvc.predict_proba(test_features)
        sv_predprob = svc.predict_proba(test_features)

        # Check that the two models have equal shape
        self.assertTupleEqual(dp_predprob.shape, sv_predprob.shape)
