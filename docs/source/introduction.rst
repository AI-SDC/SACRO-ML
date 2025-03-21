Introduction
============

What is SACRO-ML?
-----------------
SACRO-ML is a set of tools for disclosure control of trained Machine Lerning (ML) models. ML models enable the discovery of intricate relationships that a human eye, and traditional statistical methods can’t. This type of tools are powerful and are becoming increasingly popular many different fields including for medical aplications and other projects involving personal and sensitive data. Data breaches must be avoided. SACRO-ML helps to apply some mitigation startegies like the use of safemodels and estimate the risk of data leakage.

.. image:: images/ML_leakage_bee.png
    :width: 320px
    :align: center
    :height: 350px
    :alt: Data breaches of sensitive and personal data must be avoided.

When can SACRO-ML be used?
~~~~~~~~~~~~~~~~~~~~~~~~~~
- When an ML model has been trained with sensitive data and want to avoid data leakage.
- The model does not contain embedded data points.
- When the test data has not been seen by the trained model. 
- The test data must have at ideally 30 to 50% of the orginal data, and at least 20%.

What SACRO-ML is not intended for?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- For those ML models which contain embeded data. For example, instance based methods including: K-nearest neighbours (KNN), Super Vector Classifier, (SVC),  Self Organising Map (SOM), Learning Vector Quantization (LVQ), Locally Weighted Learning (LWL), Case-Based Reasoning , Gaussian Process, Kernel-based models, etc. Those model cannot be released out of the TRE as they contain embeded training data.
- Many of the deep learning models are at high risk of including data careful consideration should be applied before using SACRO-ML.
- When there is no test data or the test data has been seen by the model during the training phase.
- For any other concern that the trained model might be at risk of data breach.


What is safemodel?
------------------

The safemodel package is an open source wrapper for common machine learning
models. It is designed for use by researchers in Trusted Research Environments
(TREs) where disclosure control methods must be implemented.

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
