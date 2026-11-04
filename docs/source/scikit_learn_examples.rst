Scikit-learn Examples
=====================

This section demonstrates how to use SACRO-ML with scikit-learn models for privacy assessment.

Cancer Dataset Example
-----------------------

Training a Random Forest classification model on the breast cancer dataset and running privacy attacks.

**Training the Model:**

.. literalinclude:: ../../examples/attacks/sklearn/classification/cancer/main_train.py
   :language: python
   :linenos:
   :caption: Training Random Forest on Cancer Dataset

**Running Privacy Attacks:**

.. literalinclude:: ../../examples/attacks/sklearn/classification/cancer/main_attack.py
   :language: python
   :linenos:
   :caption: Running Privacy Attacks on Cancer Model

Nursery Dataset Example
-----------------------

Training a Random Forest classification model on the nursery dataset and assessing privacy risks.

**Training the Model:**

.. literalinclude:: ../../examples/attacks/sklearn/classification/nursery/main_train.py
   :language: python
   :linenos:
   :caption: Training Random Forest on Nursery Dataset

**Running Privacy Attacks:**

.. literalinclude:: ../../examples/attacks/sklearn/classification/nursery/main_attack.py
   :language: python
   :linenos:
   :caption: Running Privacy Attacks on Nursery Model

**Dataset Processing:**

.. literalinclude:: ../../examples/attacks/sklearn/classification/nursery/dataset.py
   :language: python
   :linenos:
   :caption: Nursery Dataset Processing

Diabetes Dataset Example
------------------------

Training a Random Forest regression model on the diabetes dataset and running privacy attacks.

**Training the Model:**

.. literalinclude:: ../../examples/attacks/sklearn/regression/diabetes/main_train.py
   :language: python
   :linenos:
   :caption: Training Random Forest Regressor on Diabetes Dataset

**Running Privacy Attacks:**

.. literalinclude:: ../../examples/attacks/sklearn/regression/diabetes/main_attack.py
   :language: python
   :linenos:
   :caption: Running Privacy Attacks on Diabetes Regression Model
