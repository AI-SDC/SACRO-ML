# Contributing

Contributions to this repository are very welcome. If you are interested in contributing, feel free to create an issue in the [issue tracking system](https://github.com/AI-SDC/SACRO-ML/issues). Alternatively, you may [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) the project and submit a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork). Please create an issue before starting any significant work so that we can discuss and understand the changes. All contributions must be made under MIT license.

## Development

Clone the repository and install the local package including all dependencies within a virtual environment:

```shell
$ git clone https://github.com/AI-SDC/SACRO-ML.git
$ cd SACRO-ML
$ pip install -e .[test]
```

Then to run the tests:

```shell
$ pytest .
```

## Repository Structure

The `CHANGELOG.md` contains a brief description of the changes made in each version.

```md
SACRO-ML
├── .github [Contains GitHub CI runner workflows]
│   └── workflows
├── docs [Contains Sphinx documentation files]
│   └── source
├── examples [Contains examples of how to run the code contained in this repository]
│   ├── notebooks [Contains example usage of the safemodel package]
│   ├── risk_examples [Contains hypothetical examples of data leakage]
│   └── user_stories [Contains user guides]
├── sacroml [Contains the sacroml source code]
│   ├── attacks [Contains a variety of privacy attacks on machine learning models]
│   ├── config [Contains code to generate configuration files]
│   └── safemodel [safemodel wrappers for common machine learning models]
└── tests [Contains unit tests]
    ├── attacks
    ├── datasets
    └── safemodel
```

## Style Guide

A [pre-commit](https://pre-commit.com) configuration [file](../tree/main/.pre-commit-config.yaml) is provided to automatically:

* Trim trailing whitespace and fix line endings;
* Check for spelling errors;
* Check and format JSON files;
* Format Python and notebooks;
* Upgrade Python syntax;
* Automatically remove unused Python imports;
* Sort Python imports.

Pre-commit can be setup as follows:

```shell
$ pip install pre-commit
```

Then to run on all files in the repository:

```shell
$ pre-commit run -a
```

Pre-commit can be configured to automatically run on every `git commit` with:

```shell
$ pre-commit install
```

## Pull Request Titles

Titles for pull requests should follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

This is a lightweight convention on top of commit messages. It provides a simple set of rules for creating an explicit, readable, and automation-friendly project history.

Individual commit messages in branches may follow an unrestricted policy, but **PR titles must follow Conventional Commits**.

### Why We Use Conventional Commit PR Titles

We require PR titles to follow the Conventional Commits format because it:

- **Enables automatic changelogs** - release notes can be generated from PR titles without manual work.
- **Clearly communicates intent** - reviewers can immediately see whether a PR is a `feat`, `fix`, `chore`, etc.
- **Improves git history navigation** - makes it easy to scan and understand changes over time.
- **Aligns with Semantic Versioning (SemVer)** - structured titles help determine version bumps automatically.
- **Supports better PR labeling & filtering** - PRs are labeled by type, making them easier to prioritise and review.
- **Flags breaking changes** - adding `!` (e.g. `feat!:`) automatically marks a PR as a breaking change.

### Format

The general structure is:

```text
<type>[optional scope]: <description>
```

Example:

```text
feat: send an email to the customer when a product is shipped
```

Types:

```text
feat — new feature
fix — bug fix
docs — documentation changes
style — formatting/styling (no code logic)
refactor — code changes without feature/bug impact
perf — performance improvements
test — adding/updating tests
build — changes to build system or dependencies
ci — changes to CI config/scripts
chore — miscellaneous maintenance tasks
revert — reverts an earlier commit
```

## Documentation

Documentation is hosted here: https://ai-sdc.github.io/SACRO-ML/

The documentation is automatically built using [Sphinx](https://www.sphinx-doc.org) and github actions.

The source files in `docs/source` are parsed/compiled into HTML files in `docs/_build`. The contents of `docs/_build` is pushed to the gh-pages branch which is then automatically deployed to the above site.

The main configuration file is `docs/source/conf.py`. Most commonly the path variable will pick up any source to document occasionally directories might need adding top the path. Please ensure to use `abspath()`

Sphinx reads the docstrings in the Python source.

It uses the numpydoc format. Your code should be documented with [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) comments.

### Quick Start

Need to get your documentation into the generated docs?

If your docstrings are in the right format, this method should work for most cases:

1. Go to `docs/source`
2. Make a copy of an rst file, e.g., `safedecisiontree.rst`
3. Edit the new file and change the title and automodule line.
4. Save the new file.
5. Edit the `index.rst` and insert the new filename (without the .rst) into the correct position in the list.
6. Save `index.rst`
7. Merge your updates to main.

### Static and Generated Content

The rst files in `docs/source/` are a mixture of static and generated content. Static content should be written in reStructuredText (rst) format.

There are lots of online tutorials for writing rst such as the [Restructured Text Primer](https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html).

### Images

It is possible to include images like this

```rst
.. image:: stars.jpg
    :width: 200px
    :align: center
    :height: 100px
    :alt: alternate text
```

### Generating Docs Locally

It is useful to be able to generate your docs locally (to check for bugs, etc.)

First install the Python dependencies with:

```shell
$ pip install -e .[doc]
```

Then run Sphinx with the following command and it should create a folder `docs/_build/html/` that will contain the html files where you can open the index.html with your web browser.

```shell
$ sphinx-build ./docs/source ./docs/_build/html/
```
