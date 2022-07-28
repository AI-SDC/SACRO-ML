Introduction
============

What is safemodel?
------------------

The safemodel package is an opensource wrapper for common machine learning 
models. It is designed for use by researchers in Trusted Research Environments (TREs) where disclosure control methods  must be implemented.

Safemodel aims to give researchers greater confidence that their models are 
more compliant with disclosure control. 

Safemodel provides feedback to the researcher through a JSON parseable 
'checkfile' report: 

.. code-block:: json

	{	
	    "researcher": "andy",
	    "model_type": "DecisionTreeClassifier",
	    "model_save_file": "unsafe.pkl",
	    "details": "WARNING: model parameters may present a disclosure risk:\n- para
	meter min_samples_leaf = 1 identified as less than the recommended min value of 
	5.",
	    "recommendation": "Do not allow release",
	    "reason": "WARNING: model parameters may present a disclosure risk:\n- param
	eter min_samples_leaf = 1 identified as less than the recommended min value of 5
	.Error: user has not called fit() method or has deleted saved values.Recommendat
	ion: Do not release."
	}

::
