Regression targets
==================

The attack entry points accept scikit-learn regressors, including pipelines,
with one or more numeric outputs per record. Construct a ``Target`` with the
fitted model and the original numeric ``y_train`` and ``y_test``. Do not encode the
numeric targets as classes or supply fabricated class probabilities.

.. code-block:: python

    from sklearn.ensemble import RandomForestRegressor
    from sacroml.attacks.target import Target
    from sacroml.attacks.worst_case_attack import WorstCaseAttack

    model = RandomForestRegressor(random_state=0).fit(X_train, y_train)
    target = Target(
        model=model, X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
    )
    attack = WorstCaseAttack(output_dir="regression_results")
    result = attack.attack(target)

Losses and membership attacks
-----------------------------

Regression loss is one MSE per record, averaging squared errors across its
output vector. For one-dimensional targets this is ``(prediction - target) ** 2``.
The generalisation gap is test MSE minus training MSE. Membership still has two
classes (member and non-member), so its AUC, TPR and FPR retain their existing
meaning even though the target model predicts numeric outputs.

* WorstCase trains its membership classifier on per-record MSE by default.
  ``include_model_correct_feature=True`` enables the Yeom comparison, which
  flags errors no greater than the training MSE, without duplicating the input.
* LiRA currently fits its shadow-model likelihoods to ``-log(MSE + 1e-16)``
  for both target and shadow models. Larger signals mean smaller errors.
  Individual reports use ``target_signal`` for both regression and classification.
  The choice between log-MSE and raw MSE is pending scientific review; neither
  transformation guarantees normally distributed signals.
* QMIA learns quantiles of negative per-record MSE instead of classification
  hinge scores. Its membership margin is the observed signal minus the learned
  quantile threshold.

These are error-based adaptations of the existing attacks. Their effectiveness
and calibration must be checked on the intended regression dataset; accepting a
regressor does not establish that a particular attack is well calibrated.
Classification score calculations are unchanged. Report metadata identifies
``target_task`` as either ``classification`` or ``regression``.

Attribute inference
-------------------

The attacker knows the original model prediction, as in the existing
classification implementation. For categorical attributes, the attack tries
each candidate and selects the one whose prediction is closest to that known
output vector, using Euclidean distance. Numerically tied candidates cause the attack to abstain.
For quantitative attributes, it searches the existing candidate grid and checks
whether all closest candidates lie within the existing protection interval.
The interval uses the absolute attribute value so negative values are handled.
The categorical confidence threshold is classification-specific; regression
uses the unique closest prediction without a probability threshold.

Structural and meta attacks
---------------------------

Structural regression reports include train/test MSE, the loss-distribution
comparison, and k-anonymity groups formed from tree leaves (for a direct decision
tree) or identical prediction vectors (for other estimators and pipelines). These are
exact groups, not tolerance-based groups for near-equal predictions.

Parameter counting supports regression trees, random forests, linear estimators
with learned coefficients and intercepts, and MLP regressors. An unsupported
parameter count produces ``null`` for the degrees-of-freedom risk.
Class disclosure, class-frequency small-group risk, and the classifier-derived
unnecessary-complexity rules are not defined for regression. These checks are
reported as ``null`` ("Not assessed" in PDF), never as evidence of safety.

MetaAttack preserves those missing assessments, including across repeated
structural attacks. Its regression structural flag reflects the available
k-anonymity indicator. Instance-based attacks continue to support their existing
regression estimators such as kNN.

Limits
------

This support covers single- and multi-output scikit-learn regression.
PyTorch regression is not covered. Existing PyTorch classification is
unchanged. Classification-specific structural criteria require a separately
agreed definition before they can assess regression models.

Use pipeline targets in memory. The existing ``Target.save()`` YAML format can
contain estimator objects from pipeline parameters that ``Target.load()`` cannot
read; that limitation also affects classification pipelines and is unchanged.
