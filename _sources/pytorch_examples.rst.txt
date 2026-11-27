PyTorch Examples
================

This section demonstrates how to use SACRO-ML with PyTorch models for privacy assessment.

PyTorch Classification Example
------------------------------

A basic example showing how to train a simple PyTorch classification model and run privacy attacks.

**Training the Model:**

.. literalinclude:: ../../examples/attacks/pytorch/classification/main_train.py
   :language: python
   :linenos:
   :caption: PyTorch Classification Model Training

Note: the training script above is included from ``examples/attacks/pytorch/classification/main_train.py``; any variables or helper functions it references (for example dataset preparation or target definitions) are defined in the example source files shown below.

**Model Definition:**

.. literalinclude:: ../../examples/attacks/pytorch/classification/model.py
   :language: python
   :linenos:
   :caption: Simple PyTorch Classification Model Architecture (from examples/attacks/pytorch/classification/model.py)

Note: this model code is included from ``examples/attacks/pytorch/classification/model.py``. Definitions used by other snippets on this page (for example a variable named ``target`` or model class definitions) come from this file.

**Running Privacy Attacks:**

.. literalinclude:: ../../examples/attacks/pytorch/classification/main_attack.py
   :language: python
   :linenos:
   :caption: Privacy Attacks on a Simple Classification PyTorch Model (from examples/attacks/pytorch/classification/main_attack.py)

Note: the attack examples are included from ``examples/attacks/pytorch/classification/main_attack.py`` and may call into the training or model files for required functions/objects.

CIFAR Dataset Example
---------------------

Advanced example using CIFAR dataset with convolutional neural networks.

**Training the Model:**

.. literalinclude:: ../../examples/attacks/pytorch/cifar/main_train.py
   :language: python
   :linenos:

Note: the training script above is included from ``examples/attacks/pytorch/cifar/main_train.py``; dataset handling and model definitions referenced below are defined in their respective files.
   :caption: CIFAR Dataset PyTorch Training (from examples/attacks/pytorch/cifar/main_train.py)

**Model Architecture:**

.. literalinclude:: ../../examples/attacks/pytorch/cifar/model.py
   :language: python
   :linenos:
   :caption: CIFAR CNN Model Architecture (from examples/attacks/pytorch/cifar/model.py)

Note: this model architecture is included from ``examples/attacks/pytorch/cifar/model.py`` and contains the network and related definitions used by the training script.

**Dataset Processing:**

.. literalinclude:: ../../examples/attacks/pytorch/cifar/dataset.py
   :language: python
   :linenos:
   :caption: CIFAR Dataset Processing (from examples/attacks/pytorch/cifar/dataset.py)

Note: dataset loading and preprocessing functions are provided in ``examples/attacks/pytorch/cifar/dataset.py``; training and evaluation snippets reference these utilities.

**Running Privacy Attacks:**

.. literalinclude:: ../../examples/attacks/pytorch/cifar/main_attack.py
   :language: python
   :linenos:
   :caption: Privacy Attacks on CIFAR Model (from examples/attacks/pytorch/cifar/main_attack.py)

Note: the attack code is taken from ``examples/attacks/pytorch/cifar/main_attack.py`` and may depend on the model and dataset code linked above.
