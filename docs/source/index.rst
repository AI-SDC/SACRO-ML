.. Documentation master file

Welcome to SACRO-ML
===================

.. toctree::
   :maxdepth: 1
   :hidden:

   support
   installation
   notebook_examples
   user_guide

An increasing body of work has shown that `machine learning <https://en.wikipedia.org/wiki/Machine_learning>`_ (ML) models may expose confidential properties of the data on which they are trained. This has resulted in a wide range of proposed attack methods with varying assumptions that exploit the model structure and/or behaviour to infer sensitive information.

The ``sacroml`` package is a collection of tools and resources for managing the `statistical disclosure control <https://en.wikipedia.org/wiki/Statistical_disclosure_control>`_ (SDC) of trained ML models. In particular, it provides:

* A **safemodel** package that extends commonly used ML models to provide *ante-hoc* SDC by assessing the theoretical risk posed by the training regime (such as hyperparameter, dataset, and architecture combinations) *before* (potentially) costly model fitting is performed. In addition, it ensures that best practice is followed with respect to privacy, e.g., using `differential privacy <https://en.wikipedia.org/wiki/Differential_privacy>`_ optimisers where available. For large models and datasets, *ante-hoc* analysis has the potential for significant time and cost savings by helping to avoid wasting resources training models that are likely to be found to be disclosive after running intensive *post-hoc* analysis.
* An **attacks** package that provides *post-hoc* SDC by assessing the empirical disclosure risk of a classification model through a variety of simulated attacks *after* training. It provides an integrated suite of attacks with a common application programming interface (API) and is designed to support the inclusion of additional state-of-the-art attacks as they become available. In addition to membership inference attacks (MIA) such as the likelihood ratio attack (`LiRA <https://doi.org/10.1109/SP46214.2022.9833649>`_) and attribute inference, the package provides novel `structural attacks <https://arxiv.org/abs/2502.09396>`_ that report cheap-to-compute metrics, which can serve as indicators of model disclosiveness after model fitting, but before needing to run more computationally expensive MIAs.
* Summaries of the results are written in a simple human-readable report.

Classification models from `scikit-learn <https://scikit-learn.org>`_ (including those implementing ``sklearn.base.BaseEstimator``) and `PyTorch <https://pytorch.org>`_ are broadly supported within the package. Some attacks can still be run if only `CSV <https://en.wikipedia.org/wiki/Comma-separated_values>`_ files of the model predicted probabilities are supplied, e.g., if the model was produced in another language.

Usage
-----

Quick-start example:

.. code-block:: python

   from sklearn.datasets import load_breast_cancer
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split

   from sacroml.attacks.likelihood_attack import LIRAAttack
   from sacroml.attacks.target import Target

   # Load dataset
   X, y = load_breast_cancer(return_X_y=True, as_frame=False)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

   # Fit model
   model = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1)
   model.fit(X_train, y_train)

   # Wrap model and data
   target = Target(
       model=model,
       dataset_name="breast cancer",
       X_train=X_train,
       y_train=y_train,
       X_test=X_test,
       y_test=y_test,
   )

   # Create an attack object and run the attack
   attack = LIRAAttack(n_shadow_models=100, output_dir="output_example")
   attack.attack(target)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Acknowledgement
===============

This work was supported by UK Research and Innovation as part of the Data and Analytics Research Environments UK (`DARE UK <https://dareuk.org.uk>`_) programme, delivered in partnership with Health Data Research UK (HDR UK) and Administrative Data Research UK (ADR UK). The specific projects were Semi-Automated Checking of Research Outputs (`SACRO <https://gtr.ukri.org/projects?ref=MC_PC_23006>`_; MC_PC_23006), Guidelines and Resources for AI Model Access from TrusTEd Research environments (`GRAIMATTER <https://gtr.ukri.org/projects?ref=MC_PC_21033>`_; MC_PC_21033), and `TREvolution <https://dareuk.org.uk/trevolution>`_ (MC_PC_24038). This project has also been supported by MRC and EPSRC (`PICTURES <https://gtr.ukri.org/projects?ref=MR%2FS010351%2F1>`_; MR/S010351/1).

.. image:: images/UK_Research_and_Innovation_logo.svg
   :width: 200

.. image:: images/health-data-research-uk-hdr-uk-logo-vector.png
   :width: 100

.. image:: images/logo_print.png
   :width: 150
