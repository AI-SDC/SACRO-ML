Dataset Handlers
================

Researcher supplied Python modules that contain a dataset class (to handle processing, splitting, etc.) that are passed to the :py:class:`sacroml.attacks.target.Target` must implement one of these abstract classes.

Scikit-learn models that use numpy arrays should implement :py:class:`SklearnDataHandler`.

PyTorch models that use DataLoaders should implement :py:class:`PyTorchDataHandler`.

-------------
API Reference
-------------

.. automodule:: sacroml.attacks.data
    :members:
