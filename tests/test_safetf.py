from safemodel.classifiers import safetf

import pytest

def test_Safe_tf_DPModel_l2_and_noise():
    """Tests user is informed this is not implemented yet"""
    with pytest.raises(NotImplementedError) :
        #with values for the l2 and noise params
        my_model = safetf.Safe_tf_DPModel(1.5,2.0)
