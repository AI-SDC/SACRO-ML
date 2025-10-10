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

**Model Definition:**

.. literalinclude:: ../../examples/pytorch/simple/model.py
   :language: python
   :linenos:
   :caption: Simple PyTorch Model Architecture

**Running Privacy Attacks:**

.. literalinclude:: ../../examples/pytorch/simple/attack_pytorch.py
   :language: python
   :linenos:
   :caption: Privacy Attacks on Simple PyTorch Model

CIFAR Dataset Example
---------------------

Advanced example using CIFAR dataset with convolutional neural networks.

**Training the Model:**

.. literalinclude:: ../../examples/pytorch/cifar/train_pytorch.py
   :language: python
   :linenos:
   :caption: CIFAR Dataset PyTorch Training

**Model Architecture:**

.. literalinclude:: ../../examples/pytorch/cifar/model.py
   :language: python
   :linenos:
   :caption: CIFAR CNN Model Architecture

**Dataset Processing:**

.. literalinclude:: ../../examples/pytorch/cifar/dataset.py
   :language: python
   :linenos:
   :caption: CIFAR Dataset Processing

**Running Privacy Attacks:**

.. literalinclude:: ../../examples/pytorch/cifar/attack_pytorch.py
   :language: python
   :linenos:
   :caption: Privacy Attacks on CIFAR Model
