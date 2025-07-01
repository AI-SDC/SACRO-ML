# Changelog

## Development

Changes:
*   Improve attribute inference attack documentation ([#340](https://github.com/AI-SDC/SACRO-ML/pull/340))
*   Refactor `Target` class - standardise use of constructor to wrap data ([#342](https://github.com/AI-SDC/SACRO-ML/pull/342))
*   Add LiRA support for multidimensional vectors such as images with multiple channels ([#343](https://github.com/AI-SDC/SACRO-ML/pull/343))
*   Remove redundant `X_orig` and `y_orig` from `Target` class ([#344](https://github.com/AI-SDC/SACRO-ML/pull/344))
*   Improve data handling for users ([#345](https://github.com/AI-SDC/SACRO-ML/pull/345))

## Version 1.3.0 (Jun 17, 2025)

Changes:
*   Abstract target model and enable support for Pytorch ([#330](https://github.com/AI-SDC/SACRO-ML/pull/330))

## Version 1.2.3 (Apr 17, 2025)

Changes:
*   Remove data preprocessing modules ([#333](https://github.com/AI-SDC/SACRO-ML/pull/333))
*   Remove tensorflow-privacy support including safekeras/safetf ([#335](https://github.com/AI-SDC/SACRO-ML/pull/335))
*   Add internal version number and move all package config to `pyproject.toml` ([#331](https://github.com/AI-SDC/SACRO-ML/pull/331))

## Version 1.2.2 (Feb 20, 2025)

Changes:
*   Fix Sphinx documentation not displaying attacks ([#305](https://github.com/AI-SDC/SACRO-ML/pull/305))
*   Fix NumPy 2.0 compatibility ([#317](https://github.com/AI-SDC/SACRO-ML/pull/317))
*   Fix AdaBoostClassifier structural test ([#318](https://github.com/AI-SDC/SACRO-ML/pull/318))
*   Fix user story examples ([#320](https://github.com/AI-SDC/SACRO-ML/pull/320), [#321](https://github.com/AI-SDC/SACRO-ML/pull/321))
*   Update acro dependency to v0.4.8 ([#322](https://github.com/AI-SDC/SACRO-ML/pull/322))
*   Add Target loading of probas from persistent storage ([#323](https://github.com/AI-SDC/SACRO-ML/pull/323))

## Version 1.2.1 (Jul 29, 2024)

Changes:
*   Rename repository from AI-SDC to SACRO-ML ([#298](https://github.com/AI-SDC/SACRO-ML/pull/298))
*   Rename package from aisdc sacroml ([#299](https://github.com/AI-SDC/SACRO-ML/pull/299))

## Version 1.2.0 (Jul 11, 2024)

Changes:
*   Add support for scikit-learn MLPClassifier ([#276](https://github.com/AI-SDC/SACRO-ML/pull/276))
*   Use default XGBoost params if not defined in structural attacks ([#277](https://github.com/AI-SDC/SACRO-ML/pull/277))
*   Clean up documentation ([#282](https://github.com/AI-SDC/SACRO-ML/pull/282))
*   Clean up repository and update packaging ([#283](https://github.com/AI-SDC/SACRO-ML/pull/283))
*   Format docstrings ([#286](https://github.com/AI-SDC/SACRO-ML/pull/286))
*   Refactor ([#284](https://github.com/AI-SDC/SACRO-ML/pull/284), [#285](https://github.com/AI-SDC/SACRO-ML/pull/285), [#287](https://github.com/AI-SDC/SACRO-ML/pull/287))
*   Add CLI and tools for generating configs; significant refactor ([#291](https://github.com/AI-SDC/SACRO-ML/pull/291))
*   Add different implementation modes for online and offline LiRA ([#281](https://github.com/AI-SDC/SACRO-ML/pull/281))

## Version 1.1.3 (Apr 26, 2024)

Changes:
*   Add built-in support for additional datasets ([#257](https://github.com/AI-SDC/SACRO-ML/pull/257))
*   Remove references to final score in outputs ([#259](https://github.com/AI-SDC/SACRO-ML/pull/259))
*   Update package dependencies: remove support for Python 3.8; add support for Python 3.11 ([#262](https://github.com/AI-SDC/SACRO-ML/pull/262))
*   Fix code coverage reporting ([#265](https://github.com/AI-SDC/SACRO-ML/pull/265))
*   Remove useless pylint suppression pragmas ([#269](https://github.com/AI-SDC/SACRO-ML/pull/269))
*   Fix axis labels in report ROC curve plot ([#270](https://github.com/AI-SDC/SACRO-ML/pull/270))

## Version 1.1.2 (Oct 30, 2023)

Changes:
*   Fix a bug related to the `rules.json` path when running from package ([#247](https://github.com/AI-SDC/SACRO-ML/pull/247))
*   Update user stories ([#247](https://github.com/AI-SDC/SACRO-ML/pull/247))

## Version 1.1.1 (Oct 19, 2023)

Changes:
*   Update notebook example paths ([#237](https://github.com/AI-SDC/SACRO-ML/pull/237))
*   Fix AdaBoostClassifier structural attack ([#242](https://github.com/AI-SDC/SACRO-ML/pull/242))
*   Move experiments module and configs to separate repository ([#229](https://github.com/AI-SDC/SACRO-ML/pull/229))

## Version 1.1.0 (Oct 11, 2023)

Changes:
*    Add automatic formatting of docstrings ([#210](https://github.com/AI-SDC/SACRO-ML/pull/210))
*    Update user stories ([#217](https://github.com/AI-SDC/SACRO-ML/pull/217))
*    Add module to run experiments with attacks and gather data ([#224](https://github.com/AI-SDC/SACRO-ML/pull/224))
*    Fix bug in report.py: error removing a file that does not exist ([#227](https://github.com/AI-SDC/SACRO-ML/pull/227))
*    Add structural attack for traditional and other risk measures ([#232](https://github.com/AI-SDC/SACRO-ML/pull/232))
*    Fix package installation for Python 3.8, 3.9, 3.10 ([#234](https://github.com/AI-SDC/SACRO-ML/pull/234))

## Version 1.0.6 (Jul 21, 2023)

Changes:
*    Update package dependencies ([#187](https://github.com/AI-SDC/SACRO-ML/pull/187))
*    Fix bug when `n_dummy_reps=0` in worst case attack ([#191](https://github.com/AI-SDC/SACRO-ML/pull/191))
*    Add ability to save target model and data to `target.json` ([#171](https://github.com/AI-SDC/SACRO-ML/pull/171), [#175](https://github.com/AI-SDC/SACRO-ML/pull/175), [#176](https://github.com/AI-SDC/SACRO-ML/pull/176), [#177](https://github.com/AI-SDC/SACRO-ML/pull/177))
*    Add safemodel SDC results to `target.json` and `attack_results.json` ([#180](https://github.com/AI-SDC/SACRO-ML/pull/180))
*    Add generalisation error to `target.json` ([#183](https://github.com/AI-SDC/SACRO-ML/pull/183))
*    Refactor attack argument handling ([#174](https://github.com/AI-SDC/SACRO-ML/pull/174))
*    Append attack outputs to a single results file ([#173](https://github.com/AI-SDC/SACRO-ML/pull/173))
*    Attack outputs written to specified folder ([#208](https://github.com/AI-SDC/SACRO-ML/pull/208))
*    Add ability to run membership inference attacks from the command line using config and target files ([#182](https://github.com/AI-SDC/SACRO-ML/pull/182))
*    Add ability to run attribute inference attacks from the command line using config and target files ([#188](https://github.com/AI-SDC/SACRO-ML/pull/188))
*    Add ability to run multiple attacks from a config file ([#200](https://github.com/AI-SDC/SACRO-ML/pull/200))
*    Add user story examples ([#194](https://github.com/AI-SDC/SACRO-ML/pull/194))
*    Improve attack formatter summary generation ([#179](https://github.com/AI-SDC/SACRO-ML/pull/179))
*    Attack formatter moves files generated for release into subfolders ([#197](https://github.com/AI-SDC/SACRO-ML/pull/197))
*    Fix a minor bug in the attack formatter ([#204](https://github.com/AI-SDC/SACRO-ML/pull/204))
*    Improve tests ([#196](https://github.com/AI-SDC/SACRO-ML/pull/196), [#199](https://github.com/AI-SDC/SACRO-ML/pull/199))

## Version 1.0.5 (Jun 5, 2023)

Changes:
*    Fix a bug calculating the number of data samples in the `Data` class ([#105](https://github.com/AI-SDC/SACRO-ML/pull/105))
*    Add a fail-fast mechanism for the worst case attack that enables the number of attack repetitions to terminate early based on a given metric and comparison operator ([#105](https://github.com/AI-SDC/SACRO-ML/pull/105))
*    Change the logging message when attack repetitions are run to 1-10 instead of 0-9 ([#105](https://github.com/AI-SDC/SACRO-ML/pull/105))
*    Add the ability to specify the number of worst case attack dummy repetitions on the command line ([#105](https://github.com/AI-SDC/SACRO-ML/pull/105))
*    Add LIRA fail-fast mechanism ([#118](https://github.com/AI-SDC/SACRO-ML/pull/118))
*    Add the ability to load LIRA attack parameters from a config file ([#118](https://github.com/AI-SDC/SACRO-ML/pull/118))
*    Add the ability to load worst case attack parameters from a config file ([#119](https://github.com/AI-SDC/SACRO-ML/pull/119))
*    Standardise the MIA attack output ([#120](https://github.com/AI-SDC/SACRO-ML/pull/120))
*    Prohibit the use of white space in report file names ([#154](https://github.com/AI-SDC/SACRO-ML/pull/154))
*    Improve the safemodel request release test ([#160](https://github.com/AI-SDC/SACRO-ML/pull/160))
*    Refactor LIRA attack tests ([#151](https://github.com/AI-SDC/SACRO-ML/pull/151))
*    Fix setting the number of LIRA shadow models from a config file ([#165](https://github.com/AI-SDC/SACRO-ML/pull/165))
*    Fix OS system calls relying on calling "python" ([#162](https://github.com/AI-SDC/SACRO-ML/pull/162))
*    Fix invalid command line argument in worst case attack example ([#164](https://github.com/AI-SDC/SACRO-ML/pull/164))
*    Add current output JSON format documentation ([#168](https://github.com/AI-SDC/SACRO-ML/pull/168))
*    Add current attack config format documentation ([#168](https://github.com/AI-SDC/SACRO-ML/pull/168))

## Version 1.0.4 (May 5, 2023)

Changes:
*    Fixed SafeRandomForestClassifier "base estimator changed" error ([#143](https://github.com/AI-SDC/SACRO-ML/pull/143))

## Version 1.0.3 (May 2, 2023)

Changes:
*    Refactored metrics ([#111](https://github.com/AI-SDC/SACRO-ML/pull/111))
*    Fixed a bug making a report when dummy reps is 0 ([#113](https://github.com/AI-SDC/SACRO-ML/pull/113))
*    Fixed safemodel JSON output ([#115](https://github.com/AI-SDC/SACRO-ML/pull/115))
*    Added a module to produce recommendations from attack JSON output ([#116](https://github.com/AI-SDC/SACRO-ML/pull/116))
*    Disabled non-default report logs ([#123](https://github.com/AI-SDC/SACRO-ML/pull/123))
*    Fixed a minor bug in worst case example ([#124](https://github.com/AI-SDC/SACRO-ML/pull/124))

## Version 1.0.2 (Feb 27, 2023)

Changes:
*    Added support for Python 3.8, 3.9 and 3.10 and update requirements.
*    Fixed documentation links to notebooks and added SafeSVC.
*    Added option to include target model error into attacks as a feature.

## Version 1.0.1 (Nov 16, 2022)

Changes:
*    Increased test coverage.
*    Packaged for PyPI.

## Version 1.0.0 (Sep 14, 2022)

First version.
