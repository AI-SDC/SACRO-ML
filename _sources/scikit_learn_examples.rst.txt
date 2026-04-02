Scikit-learn Examples
=====================

This section demonstrates how to use SACRO-ML with scikit-learn models for privacy assessment.

Cancer Dataset Example
-----------------------

Training a Random Forest model on the breast cancer dataset and running privacy attacks.

**Training the Model:**

.. literalinclude:: ../../examples/sklearn/cancer/train_rf_cancer.py
   :language: python
   :linenos:
   :caption: Training Random Forest on Cancer Dataset

**Running Privacy Attacks:**

.. literalinclude:: ../../examples/sklearn/cancer/attack_rf_cancer.py
   :language: python
   :linenos:
   :caption: Running Privacy Attacks on Cancer Model

Nursery Dataset Example
-----------------------

Training a Random Forest model on the nursery dataset and assessing privacy risks.

**Training the Model:**

.. literalinclude:: ../../examples/sklearn/nursery/train_rf_nursery.py
   :language: python
   :linenos:
   :caption: Training Random Forest on Nursery Dataset

**Running Privacy Attacks:**

.. literalinclude:: ../../examples/sklearn/nursery/attack_rf_nursery.py
   :language: python
   :linenos:
   :caption: Running Privacy Attacks on Nursery Model

**Dataset Processing:**

.. literalinclude:: ../../examples/sklearn/nursery/dataset.py
   :language: python
   :linenos:
   :caption: Nursery Dataset Processing
