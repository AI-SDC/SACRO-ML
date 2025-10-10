Unsupported Model Examples
===========================

This section shows how to work with models that are not directly supported by SACRO-ML's built-in wrappers.

Working with Unsupported Models
--------------------------------

When working with models not directly supported by SACRO-ML, you can still perform privacy attacks by using CSV outputs or custom model wrappers.

**Training an Unsupported Model:**

.. literalinclude:: ../../examples/unsupported/train.py
   :language: python
   :linenos:
   :caption: Training Unsupported Model

**Running Attacks on Unsupported Models:**

.. literalinclude:: ../../examples/unsupported/attack.py
   :language: python
   :linenos:
   :caption: Attacking Unsupported Model

These examples demonstrate how to adapt SACRO-ML for use with any machine learning framework by using model predictions as CSV files or creating custom model interfaces.
