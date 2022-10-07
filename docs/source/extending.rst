Extending SafeModel
===================

Modular Design
--------------

The safemodel package is an opensource wrapper for common machine learning
models. It is designed to be modular and can be extended for use with other
models. Code comments should be in the numpydoc format so that they are rendered
by the automatic sphinx documentation

The main steps needed to implement a new model are:

# Copy the new_model_template.py
# Define a safer class inheriting SafeModel and the Basic (SkLearn) model
# Update the __init__ method with ignore_items and examine_separately items
# Add checks for any unusual data structures
# Override the fit() function
# Update Sphinx documentation
# Write pytests to confirm core functionality
# Include any optional helper functions

Copy The Template
-----------------

.. code-block:: shell

		cp new_model_template xgboost.py

Define the Safer Class
----------------------

.. code-block:: python

	class SafeGradientBoosting(SafeModel, GradientBoostingClassifier):
		"""Privacy protected GradientBoostingClassifier."""


Update rules.json file
----------------------

The rules.json file is used to define safe limits for pearameters.
The file is written in JSON (JavaScript Object Notation) and can be extended.
to define safe limits for parameters of newly implemented models.

Update the __init__ method with paramnames, ignore_items, and examine_separately items
--------------------------------------------------------------------------------------

Code for a new class needs to reflect is the contents of the list self.basemodel_paramnames.

.. code-block:: python

	class SafeModelToMakeSafe(SafeModel, GradientBoostingClassifier):
		"""Privacy protected XGBoost."""

    	def __init__(self, **kwargs: Any) -> None:
        	"""Creates model and applies constraints to params"""
        	SafeModel.__init__(self)

        	self.basemodel_paramnames=[
            	'edit','this','list','to',
            	'contain','just','the','valid','parameters',
            	'for','the','class',
            	'you ','are','creating','a'
            	'safe','wrapper','version','of']

        	the_kwds=dict()
        	for key,val in kwargs.items():
            	if key in self.basemodel_paramnames:
                	the_kwds[key]=val
        	ModelToMakeSafer.__init__(self, **the_kwds)
        	self.model_type: str = "ModelToMakeSafer"
        	super().preliminary_check(apply_constraints=True, verbose=True)
        	self.ignore_items = [
            	"model_save_file",
            	"ignore_items",
            	"base_estimator_",
        	]
        	self.examine_seperately_items = ["base_estimator", "estimators_"]

For sklearn models this list can be extracted from the sklearn man page for the new model. For example,
Saferandomforest defines the valid paramnames as:

.. code-block:: python

	def __init__(self, **kwargs: Any) -> None:
        	"""Creates model and applies constraints to params"""
        	SafeModel.__init__(self)
        	self.basemodel_paramnames=[
            	'n_estimators','criterion','max_depth','min_samples_split',
            	'min_samples_leaf','min_weight_fraction_leaf','max_features',
            	'max_leaf_nodes','min_impurity_decrease','bootstrap',
            	'oob_score','n_jobs','random_state','verbose'
            	'warm_start','class_weight','ccp_alpha','max_samples']

Add checks for any unusual data structures
------------------------------------------

Some models may have unusual datastructures.
Care should be taken to ensure that these are not changed after the fit() method
is called.

Examples of unusual datastructures are:
Lists are handled in the safemodel base class.
Decision Trees handled in safedecisiontree.py and saferandomforest.py

.. code-block:: python

	class SafeGradientBoosting(SafeModel, GradientBoostingClassifier):
		"""Privacy protected GradientBoostingClassifier."""

Override the fit() function
---------------------------

.. code-block:: python


	def fit(self, x: np.ndarray, y: np.ndarray) -> None:
		"""Do fit and then store model dict"""
		super().fit(x, y)
		self.k_anonymity = self.get_k_anonymity(x)
		self.saved_model = copy.deepcopy(self.__dict__)

Update Sphinx documentation
----------------------------

In the Sphinx docs/source directory make a copy of an existing .rst file
it the .rst to reflect the newly implemented class. Then you must update the
index.rst file by to include the new .rst file, although the extension is
not required. E.g. saferandomforest links in saferandomforest.rst

.. code-block:: shell

	cd docs
	cp saferandomforest.rst xgboost.rst
	edit xgboost.rst
	edit index.rst

Write pytests to confirm core functionality
--------------------------------------------

Write pytests to confirm the corefunctionality.
Example test suites can be found in AI-SDC/tests/

Include any optional helper functions
-------------------------------------

Depending on the model being implemented one or more helper functions or
methods may be required. For example there are may helpfunctions in
safekeras.py that help with the the specifics of neural networks.

.. code-block:: python

	def same_weights(m1: Any, m2: Any) -> Tuple[bool, str]:
	if len(m1.layers) != len(m2.layers):
		return False, "different numbers of layers"
	numlayers = len(m1.layers)
	for layer in range(numlayers):
		m1layer = m1.layers[layer].get_weights()
		m2layer = m2.layers[layer].get_weights()
        if len(m1layer) != len(m2layer):
            return False, f"layer {layer} not the same size."
        for dim in range(len(m1layer)):
            m1d = m2layer[dim]
            m2d = m2layer[dim]
            # print(type(m1d), m1d.shape)
            if not np.array_equal(m1d, m2d):
                return False, f"dimension {dim} of layer {layer} differs"
	    return True, "weights match"
