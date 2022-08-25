Extending SafeModel
===================

Modular Design
--------------

The safemodel package is an opensource wrapper for common machine learning
models. It is designed to be modular and can be extended for use with other
models.

The main steps needed to implement a new model are:

# Copy the new_model_template.py
# Define a safer class inheriting SafeModel and the Basic (SkLearn) model
# Update the __init__ method with ignore_items and examine_separately items
# Add checks for any unusual data structures
# Override the fit() function
# Comment the code using numydoc format
# Update Sphinx documentation
# Write pytests to confirm core functionality 
# Include any optional helper functions

Copy The Template
-----------------

.. code-block:: shell

	{
		cp new_model_template xgboost.py
	}

::

Define the Safer Class
----------------------

.. code-block:: python

	{
	class SafeGradientBoosting(SafeModel, GradientBoostingClassifier):
		"""Privacy protected GradientBoostingClassifier."""

	}

::

   
Update the __init__ method with ignore_items and examine_separately items
-------------------------------------------------------------------------

.. code-block:: python

	{
	class SafeModelToMakeSafe(SafeModel, GradientBoostingClassifier):
	"""Privacy protected XGBoost."""

	def __init__(self, **kwargs: Any) -> None:
		"""Creates model and applies constraints to params"""
		SafeModel.__init__(self)
		GradientBoostingClassifier.__init__(self, **kwargs)
		self.model_type: str = "GradientBoostingClassifier"
		super().preliminary_check(apply_constraints=True, verbose=True)
		self.ignore_items = [
		    "model_save_file",
                    "ignore_items",
                    "base_estimator_",
		]
		self.examine_seperately_items = ["base_estimator", "estimators_"]


	}

::


Add checks for any unusual data structures
------------------------------------------


.. code-block:: python

	{
	class SafeGradientBoosting(SafeModel, GradientBoostingClassifier):
		"""Privacy protected GradientBoostingClassifier."""

	}

Override the fit() function
---------------------------

.. code-block:: python

	{
	class SafeGradientBoosting(SafeModel, GradientBoostingClassifier):
		"""Privacy protected GradientBoostingClassifier."""

	}

Comment the code using numydoc format
--------------------------------------

.. code-block:: python

	{
	class SafeGradientBoosting(SafeModel, GradientBoostingClassifier):
		"""Privacy protected GradientBoostingClassifier."""

	}

Update Sphinx documentation
----------------------------


.. code-block:: python

	{
	class SafeGradientBoosting(SafeModel, GradientBoostingClassifier):
		"""Privacy protected GradientBoostingClassifier."""

	}
	
Write pytests to confirm core functionality 
--------------------------------------------


.. code-block:: python

	{
	class SafeGradientBoosting(SafeModel, GradientBoostingClassifier):
		"""Privacy protected GradientBoostingClassifier."""

	}
	
Include any optional helper functions
-------------------------------------

.. code-block:: python

	{
	class SafeGradientBoosting(SafeModel, GradientBoostingClassifier):
		"""Privacy protected GradientBoostingClassifier."""

	}
