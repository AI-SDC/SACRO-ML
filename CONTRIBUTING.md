# General guidance for contributors


# Adding your files to the automatically created documentation website
Automatic Documentation
=======================

The documentation is automatically built using sphinx and github actions.

The source files in docs/source are parsed/compiled into HTML files in docs/_build.
The contents of docs/_build is pushed to the gh-pages branch which is then automatically 
deployed to the github.io site. 

The main configuration file is docs/source/conf.py
Most commonly the path variable will pick up any source to document
occassionaly directories might need adding top the path. Please ensure to use abspath()

Sphinx reads the docstrings in the python source.

It uses the numpydoc format. Your code should be documented with numpydoc comments.
[NumpyDoc](https://numpydoc.readthedocs.io/en/latest/format.html). 

Quick Start
===========

Need to get your documentation into the generated docs?
If your DocStrings are in the right format, this method should work for most cases:

1. Go to docs/source
2. Make a copy of an rst file (E.g. (safedecisiontree.rst)
3. edit the new file and change the title and automodule line.
```
Data Interface
==============
 
An example Python Notebook is available  `Here <https://github.com/jim-smith/GRAIMatter/blob/main/WP2/wrapper/wrapper-concept.ipynb>`_
 
.. automodule:: preprocessing.loaders
   :members:
```

4. Save the new file
5. Edit the index.rst and insert the new filename(without the .rst) into the correct position in the list.
```
.. toctree::
   :maxdepth: 2
   :caption: Contents:
 
   introduction
   attacks
   safemodel
   safedecisiontree
   saferandomforest
   safekeras
   datainterface
```

6. Save index.rst
7. Push your updates to main

DocStrings
==========

An example docstring from
the safemodel source is below: 

```
class SafeModel:
      """Privacy protected model base class.
      Attributes
      ----------
      model_type: string
            A string describing the type of model. Default is "None".
      model:
            The Machine Learning Model. 
      saved_model:
            A saved copy of the Machine Learning Model used for comparisson.
      ignore_items: list
            A list of items to ignore when comparing the model with the 
            saved_model.
      examine_separately_items: list
            A list of items to examine separately. These items are more 
            complex datastructures that cannot be compared directly.
      filename: string
            A filename to save the model. 
      researcher: string
            The researcher user-id used for logging 
      Notes
      -----
      Examples
      --------
      >>> safeRFModel = SafeRandomForestClassifier()
      >>> safeRFModel.fit(X, y)
      >>> safeRFModel.save(name="safe.pkl")
      >>> safeRFModel.preliminary_check()
      >>> safeRFModel.request_release(filename="safe.pkl")
      WARNING: model parameters may present a disclosure risk:
      - parameter min_samples_leaf = 1 identified as less than the recommended min value of 5.
      Changed parameter min_samples_leaf = 5.
      Model parameters are within recommended ranges.
      """
```

Static and Generated Content
============================

The .rst files in docs/source/ are a mixture of static and generated content
Static content should be written in ReStructuredText (.rst) format.

A short primer  

[Restructured Text Primer](https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html#introduction)

Automatic code documentation from the docstrings uses sphinx directives in the .rst files like this:

```
.. automodule:: safemodel.classifiers.safedecisiontreeclassifier
   :members:
```
Images
------

It is possible to include images like this

```
 
.. image:: stars.jpg
    :width: 200px
    :align: center
    :height: 100px
    :alt: alternate text
```

Generating docs locally
=======================

It is useful to be able to generate your docs locally (to check for bugs etc)

In linux, navigate to the `docs` folder and then issue the command
`make html`

In windows, navigate to the `docs` folder than then issue the command
`sphinx-build source _build`

The generated html will be in the folder `docs/_build` and can be opened in any browser.



