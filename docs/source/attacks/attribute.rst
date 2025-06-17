Attribute Attack
================

The attribute inference attack assumes that the attacker has access to a record missing the value for one attribute, and measuring whether the trained model allows more (and more accurate) predicted completions for the training set than it does for the test set. An exhaustive search of the target model's predictive confidence is performed for all possible values that complete the record (discretised for continuous attributes). The attack model then makes a prediction if one missing value (categorical) or a single unbroken range of values (continuous) leads to the highest target confidence, which is above a user-defined threshold; otherwise it reports *don't know*.

The attack computes an upper bound on the fraction of records that are vulnerable, i.e., where the attack makes a correct prediction, and reports the Attribute Risk Ratio, ARR(*a*): the ratio of training and test set proportions for each attribute *a*. The attack is considered accurate if the target model's predicted label :math:`l^*` for the record with a single missing value is the same as for the actual record value :math:`l` (categorical) or the range of values yielding the same target confidence lies within :math:`l\pm10\%` (continuous). This latter condition mirrors the protection limits commonly used in cell suppression algorithms; see, for example, :cite:t:`Smith:2012`. The ARR metric recognises that any useful trained model contains some generalisable information and so only considers the model to be leaking privacy if ARR(*a*)>1. It also recognises that not all attributes will be considered equally disclosive, and so enables a discussion between TRE staff and researchers.

-----
Usage
-----

To run the attribute attack, in addition to the usual processed data splits, the feature encoding and the original unprocessed data must be included within the :py:class:`sacroml.attacks.target.Target` that is passed to the attribute attack object.

See the examples:

* `Training <https://www.github.com/ai-sdc/sacro-ml/tree/main/examples/train_rf_nursery.py>`_ a model and including all required information.
* `Running <https://www.github.com/ai-sdc/sacro-ml/tree/main/examples/attack_attribute.py>`_ an attribute inference attack programmatically.

----------
References
----------

.. bibliography::

-------------
API Reference
-------------

.. automodule:: sacroml.attacks.attribute_attack
    :members:
