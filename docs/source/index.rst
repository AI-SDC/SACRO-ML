.. raw:: html

   <div style="text-align: left; margin: 20px 0;">
      <img src="./_static/SACRO_Logo_final.png" alt="SACRO Logo" width="100" style="background: transparent !important; border: none;" />
   </div>


========================================
Welcome to the AI-SDC family of tools
========================================

Our tools are designed to help researchers assess the privacy disclosure risks of their outputs, including tables, plots, statistical models, and trained machine learning models


.. toctree::
   :maxdepth: 1
   :hidden:

   introduction
   support
   installation
   examples
   user_guide

.. grid:: 2

    .. grid-item-card:: ACRO (Python)
        :link: https://sacro-tools.org/ACRO/introduction.html
        :link-type: url
        :shadow: md
        :class-header: bg-info

        **Statistical Disclosure Control for Python**

        Tools for the Semi-Automatic Checking of Research Outputs. Drop-in replacements for common analysis commands with built-in privacy protection.

        +++

        :bdg-info:`Statistical Analysis` `Visit ACRO Docs →`

    .. grid-item-card:: SACRO-ML
        :link: introduction
        :link-type: doc
        :shadow: md
        :class-header: bg-primary

        **Machine Learning Privacy Tools**

        Collection of tools and resources for managing the statistical disclosure control of trained machine learning models.

        +++

        :bdg-primary:`Current Documentation Focus` :doc:`Get Started → <introduction>`

.. grid:: 2

    .. grid-item-card:: ACRO-R
        :link: https://jessuwe.github.io/ACRO/introduction.html
        :link-type: url
        :shadow: md
        :class-header: bg-success

        **R Package Integration**

        R-language interface for the Python ACRO library, providing familiar R syntax for statistical disclosure control.

        +++

        :bdg-success:`R Integration` `Explore ACRO-R →`

    .. grid-item-card:: SACRO-Viewer
        :link: https://jessuwe.github.io/SACRO-Viewer/introduction.html
        :link-type: url
        :shadow: md
        :class-header: bg-warning

        **Graphical User Interface**

        A graphical user interface for fast, secure and effective output checking, which can work in any TRE (Trusted Research Environment).

        +++

        :bdg-warning:`GUI Tool` `View Docs →`

SACRO-ML: Machine Learning Privacy Tools
=========================================

SACRO-ML is a free and open source collection of tools and resources for managing the statistical disclosure control (SDC) of trained machine learning models. It provides both ante-hoc and post-hoc privacy assessment capabilities for researchers working with ML models in secure data environments.

.. note::
   **New in v1.4.0:** Enhanced support for PyTorch models and improved structural attack capabilities.


Getting Started
===============

.. grid:: 3

    .. grid-item-card:: Install
        :link: installation
        :link-type: doc
        :class-header: bg-light

        Get SACRO-ML installed and configured in your environment

    .. grid-item-card:: Learn
        :link: examples
        :link-type: doc
        :class-header: bg-light

        Explore comprehensive examples for all frameworks and use cases

    .. grid-item-card:: Reference
        :link: attacks/index
        :link-type: doc
        :class-header: bg-light

        Complete API documentation and attack reference



Community and Support
=====================

.. grid:: 2

    .. grid-item-card:: Get Help
        :class-header: bg-light

        * `GitHub Issues <https://github.com/AI-SDC/SACRO-ML/issues>`_
        * `Discussion Forum <https://github.com/AI-SDC/SACRO-ML/discussions>`_
        * Email: sacro.contact@uwe.ac.uk

    .. grid-item-card:: Contribute
        :class-header: bg-light

        * :doc:`Contributing Guide <contributing>`
        * `Source Code <https://github.com/AI-SDC/SACRO-ML>`_
        * `Report Issues <https://github.com/AI-SDC/SACRO-ML/issues/new>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Acknowledgement
===============

This work was supported by UK Research and Innovation as part of the Data and Analytics Research Environments UK (`DARE UK <https://dareuk.org.uk>`_) programme, delivered in partnership with Health Data Research UK (HDR UK) and Administrative Data Research UK (ADR UK). The specific projects were Semi-Automated Checking of Research Outputs (`SACRO <https://gtr.ukri.org/projects?ref=MC_PC_23006>`_; MC_PC_23006), Guidelines and Resources for AI Model Access from TrusTEd Research environments (`GRAIMATTER <https://gtr.ukri.org/projects?ref=MC_PC_21033>`_; MC_PC_21033), and `TREvolution <https://dareuk.org.uk/trevolution>`_ (MC_PC_24038). This project has also been supported by MRC and EPSRC (`PICTURES <https://gtr.ukri.org/projects?ref=MR%2FS010351%2F1>`_; MR/S010351/1).

.. image:: images/UK_Research_and_Innovation_logo.svg
   :width: 200

.. image:: images/health-data-research-uk-hdr-uk-logo-vector.png
   :width: 100

.. image:: images/logo_print.png
   :width: 150
