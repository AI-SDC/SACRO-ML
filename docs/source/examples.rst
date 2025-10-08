Examples
========

This section contains comprehensive examples demonstrating how to use SACRO-ML for various machine learning scenarios and frameworks.

Jupyter Notebook Examples
==========================

Interactive notebook examples for common use cases:

.. toctree::
   :maxdepth: 1

   notebooks/example-notebook-decisiontree
   notebooks/example-notebook-randomforest
   notebooks/example-notebook-SVC

Scikit-learn Examples
=====================

.. include:: ../../examples/sklearn/README.md
   :parser: myst_parser.sphinx_

Cancer Dataset Example
----------------------

.. literalinclude:: ../../examples/sklearn/cancer/train_rf_cancer.py
   :language: python
   :linenos:
   :caption: Training Random Forest on Cancer Dataset

.. literalinclude:: ../../examples/sklearn/cancer/attack_rf_cancer.py
   :language: python
   :linenos:
   :caption: Running Privacy Attacks on Cancer Model

Nursery Dataset Example
-----------------------

.. literalinclude:: ../../examples/sklearn/nursery/train_rf_nursery.py
   :language: python
   :linenos:
   :caption: Training Random Forest on Nursery Dataset

.. literalinclude:: ../../examples/sklearn/nursery/attack_rf_nursery.py
   :language: python
   :linenos:
   :caption: Running Privacy Attacks on Nursery Model

PyTorch Examples
================

.. include:: ../../examples/pytorch/README.md
   :parser: myst_parser.sphinx_

Simple PyTorch Example
----------------------

.. literalinclude:: ../../examples/pytorch/simple/train_pytorch.py
   :language: python
   :linenos:
   :caption: Simple PyTorch Model Training

.. literalinclude:: ../../examples/pytorch/simple/attack_pytorch.py
   :language: python
   :linenos:
   :caption: Privacy Attacks on Simple PyTorch Model

CIFAR PyTorch Example
---------------------

.. literalinclude:: ../../examples/pytorch/cifar/train_pytorch.py
   :language: python
   :linenos:
   :caption: CIFAR Dataset PyTorch Training

.. literalinclude:: ../../examples/pytorch/cifar/attack_pytorch.py
   :language: python
   :linenos:
   :caption: Privacy Attacks on CIFAR Model

Risk Assessment Examples
========================

.. include:: ../../examples/risk_examples/README.md
   :parser: myst_parser.sphinx_

Python Risk Examples
--------------------

.. toctree::
   :maxdepth: 1

   ../../examples/risk_examples/python/membership_inference_cancer
   ../../examples/risk_examples/python/attribute_inference_cancer
   ../../examples/risk_examples/python/instance_based_mimic

SafeModel Examples
==================

.. literalinclude:: ../../examples/safemodel/safemodel.py
   :language: python
   :linenos:
   :caption: SafeModel Usage Example

Unsupported Model Examples
===========================

.. include:: ../../examples/unsupported/README.md
   :parser: myst_parser.sphinx_

.. literalinclude:: ../../examples/unsupported/train.py
   :language: python
   :linenos:
   :caption: Training Unsupported Model

.. literalinclude:: ../../examples/unsupported/attack.py
   :language: python
   :linenos:
   :caption: Attacking Unsupported Model

User Stories
============

.. include:: ../../examples/user_stories/README.md
   :parser: myst_parser.sphinx_

User Story 1: Basic Model Training and Attack
----------------------------------------------

.. literalinclude:: ../../examples/user_stories/user_story_1/user_story_1_researcher_template.py
   :language: python
   :linenos:
   :caption: User Story 1 - Researcher Template

.. literalinclude:: ../../examples/user_stories/user_story_1/user_story_1_tre.py
   :language: python
   :linenos:
   :caption: User Story 1 - TRE Implementation

User Story 2: Data Processing and Privacy Assessment
-----------------------------------------------------

.. literalinclude:: ../../examples/user_stories/user_story_2/user_story_2_researcher_template.py
   :language: python
   :linenos:
   :caption: User Story 2 - Researcher Template

.. literalinclude:: ../../examples/user_stories/user_story_2/user_story_2_tre.py
   :language: python
   :linenos:
   :caption: User Story 2 - TRE Implementation

User Story 3: Advanced Privacy Analysis
----------------------------------------

.. literalinclude:: ../../examples/user_stories/user_story_3/user_story_3_researcher_template.py
   :language: python
   :linenos:
   :caption: User Story 3 - Researcher Template

.. literalinclude:: ../../examples/user_stories/user_story_3/user_story_3_tre.py
   :language: python
   :linenos:
   :caption: User Story 3 - TRE Implementation