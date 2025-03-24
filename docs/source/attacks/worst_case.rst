Worst Case Attack
=================

The main objective of a membership inference attack (MIA) to predict if a specific data point was part or not of the training data of a given model.

The Worst-Case is a white box MIA scenario, and it does not need any shadow model. It is described in `Rezaei`_  as the easiest possible for the attacker. To perform a MIA, a new binary set of data (member, non-member of target train data) was created containing the predicted probabilities of the training and test data by the target model. A Random Forest Classifier was fitted with half of this new set.

This scenario was not supposed to simulate a realistic attack (if the attacker has access to the data, they do not need to attack) but instead to assess whether there were potential vulnerabilities in the model that could potentially be leveraged by an attacker. This can give a good estimation of the maximum capability of an attacker to succeed.

In some cases, the risk of data leakage could be overestimated, but it does guarantee (as much as possible) that any ML model allowed out of a TRE is safe. At the same time, it’s easy to implement (see figure below).

This attack is, however, evaluated using average-case “accuracy” metrics that fail to characterize whether the attack can confidently identify any members of the training set.

.. image:: ../images/WorstCase_diagram.png
    :width: 350px
    :align: center
    :height: 350px
    :alt: Data breaches of sensitive and personal data must be avoided.

.. automodule:: sacroml.attacks.worst_case_attack
    :members:

.. _Rezaei: https://openaccess.thecvf.com/content/CVPR2021/papers/Rezaei_On_the_Difficulty_of_Membership_Inference_Attacks_CVPR_2021_paper.pdf
