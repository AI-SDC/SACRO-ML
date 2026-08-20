PyTorch Examples
================

This section demonstrates how to use SACRO-ML with PyTorch models for privacy assessment.

Simple PyTorch Example
-----------------------

A basic example showing how to train a simple PyTorch model and run privacy attacks.

**Training the Model:**

.. literalinclude:: ../../examples/pytorch/simple/train_pytorch.py
   :language: python
   :linenos:
   :caption: Simple PyTorch Model Training

Note: the training script above is included from ``examples/pytorch/simple/train_pytorch.py``; any variables or helper functions it references (for example dataset preparation or target definitions) are defined in the example source files shown below.

**Model Definition:**

.. literalinclude:: ../../examples/pytorch/simple/model.py
   :language: python
   :linenos:
   :caption: Simple PyTorch Model Architecture (from examples/pytorch/simple/model.py)

Note: this model code is included from ``examples/pytorch/simple/model.py``. Definitions used by other snippets on this page (for example a variable named ``target`` or model class definitions) come from this file.

**Running Privacy Attacks:**

.. literalinclude:: ../../examples/pytorch/simple/attack_pytorch.py
   :language: python
   :linenos:
   :caption: Privacy Attacks on Simple PyTorch Model (from examples/pytorch/simple/attack_pytorch.py)

Note: the attack examples are included from ``examples/pytorch/simple/attack_pytorch.py`` and may call into the training or model files for required functions/objects.

CIFAR Dataset Example
---------------------

Advanced example using CIFAR dataset with convolutional neural networks.

**Training the Model:**

.. literalinclude:: ../../examples/pytorch/cifar/train_pytorch.py
   :language: python
   :linenos:

Note: the training script above is included from ``examples/pytorch/cifar/train_pytorch.py``; dataset handling and model definitions referenced below are defined in their respective files.
   :caption: CIFAR Dataset PyTorch Training (from examples/pytorch/cifar/train_pytorch.py)

**Model Architecture:**

.. literalinclude:: ../../examples/pytorch/cifar/model.py
   :language: python
   :linenos:
   :caption: CIFAR CNN Model Architecture (from examples/pytorch/cifar/model.py)

Note: this model architecture is included from ``examples/pytorch/cifar/model.py`` and contains the network and related definitions used by the training script.

**Dataset Processing:**

.. literalinclude:: ../../examples/pytorch/cifar/dataset.py
   :language: python
   :linenos:
   :caption: CIFAR Dataset Processing (from examples/pytorch/cifar/dataset.py)

Note: dataset loading and preprocessing functions are provided in ``examples/pytorch/cifar/dataset.py``; training and evaluation snippets reference these utilities.

**Running Privacy Attacks:**

.. literalinclude:: ../../examples/pytorch/cifar/attack_pytorch.py
   :language: python
   :linenos:
   :caption: Privacy Attacks on CIFAR Model (from examples/pytorch/cifar/attack_pytorch.py)

Note: the attack code is taken from ``examples/pytorch/cifar/attack_pytorch.py`` and may depend on the model and dataset code linked above.
