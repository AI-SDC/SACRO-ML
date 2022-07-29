# General guidance for contributors


# Adding your files to the automatically created documentation website
Automatic Documentation
=======================

The documentation is automatically build using sphinx and github actions.

The source files in docs/source are parsed/compiled into HTML files in docs/_build.
The contents of docs/_build is pushed to the gh-pages branch which is then automatically 
deployed to the github.io site. 

The main configuration file is docs/source/conf.py
Most commonly the path variable will pick up any source to document
occassionaly directories might need adding top the path. Please ensure to use abspath()

Sphinx reads the docstrings in the python source.

It uses the numpydoc format. Your code should be documented with numpydoc comments.
NumpyDoc]https://numpydoc.readthedocs.io/en/latest/format.html) 

Static and Generated Content
============================

The .rst files in docs/source/ are a mixture of static and generated content
Static content should be written in ReStructuredText (.rst) format.

A short primer  

[Restructured Text Primer](https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html#introduction)

Automatic code documentation uses sphinx directives like this:

.. automodule:: safemodel.classifiers.safedecisiontreeclassifier
   :members:

Images
------
 

 
.. image:: stars.jpg
    :width: 200px
    :align: center
    :height: 100px
    :alt: alternate text





