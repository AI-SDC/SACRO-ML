# Changelog

## Development

Changes:
*    Add ability to save target model and data to `target.json` ([#171](https://github.com/AI-SDC/AI-SDC/pull/171), [#175](https://github.com/AI-SDC/AI-SDC/pull/175), [#176](https://github.com/AI-SDC/AI-SDC/pull/176), [#177](https://github.com/AI-SDC/AI-SDC/pull/177))
*    Add safemodel SDC results to `target.json` and `attack_results.json` ([#180](https://github.com/AI-SDC/AI-SDC/pull/180))
*    Add generalisation error to `target.json` ([#183](https://github.com/AI-SDC/AI-SDC/pull/183))
*    Update package dependencies ([#187](https://github.com/AI-SDC/AI-SDC/pull/187))
*    Append attack outputs to a single results file ([#173](https://github.com/AI-SDC/AI-SDC/pull/173))
*    Refactor attack argument handling ([#174](https://github.com/AI-SDC/AI-SDC/pull/174))
*    Add ability to run membership inference attacks from the command line using config and target files ([#182](https://github.com/AI-SDC/AI-SDC/pull/182))
*    Add ability to run attribute inference attacks from the command line using config and target files ([#188](https://github.com/AI-SDC/AI-SDC/pull/188))
*    Add ability to run multiple attacks from a config file ([#200](https://github.com/AI-SDC/AI-SDC/pull/200))
*    Fix bug when `n_dummy_reps=0` in worst case attack ([#191](https://github.com/AI-SDC/AI-SDC/pull/191))
*    Add user story examples ([#194](https://github.com/AI-SDC/AI-SDC/pull/194))
*    Improve attack formatter summary generation ([#179](https://github.com/AI-SDC/AI-SDC/pull/179))
*    Attack formatter moves files generated for release into subfolders ([#197](https://github.com/AI-SDC/AI-SDC/pull/197))
*    Fix a minor bug in the attack formatter ([#204](https://github.com/AI-SDC/AI-SDC/pull/204))
*    Improve tests ([#196](https://github.com/AI-SDC/AI-SDC/pull/196), [#199](https://github.com/AI-SDC/AI-SDC/pull/199), [#205](https://github.com/AI-SDC/AI-SDC/pull/205))

## Version 1.0.5 (Jun 5, 2023)

Changes:
*    Fix a bug calculating the number of data samples in the `Data` class ([#105](https://github.com/AI-SDC/AI-SDC/pull/105))
*    Add a fail-fast mechanism for the worst case attack that enables the number of attack repetitions to terminate early based on a given metric and comparison operator ([#105](https://github.com/AI-SDC/AI-SDC/pull/105))
*    Change the logging message when attack repetitions are run to 1-10 instead of 0-9 ([#105](https://github.com/AI-SDC/AI-SDC/pull/105))
*    Add the ability to specify the number of worst case attack dummy repetitions on the command line ([#105](https://github.com/AI-SDC/AI-SDC/pull/105))
*    Add LIRA fail-fast mechanism ([#118](https://github.com/AI-SDC/AI-SDC/pull/118))
*    Add the ability to load LIRA attack parameters from a config file ([#118](https://github.com/AI-SDC/AI-SDC/pull/118))
*    Add the ability to load worst case attack parameters from a config file ([#119](https://github.com/AI-SDC/AI-SDC/pull/119))
*    Standardise the MIA attack output ([#120](https://github.com/AI-SDC/AI-SDC/pull/120))
*    Prohibit the use of white space in report file names ([#154](https://github.com/AI-SDC/AI-SDC/pull/154))
*    Improve the safemodel request release test ([#160](https://github.com/AI-SDC/AI-SDC/pull/160))
*    Refactor LIRA attack tests ([#151](https://github.com/AI-SDC/AI-SDC/pull/151))
*    Fix setting the number of LIRA shadow models from a config file ([#165](https://github.com/AI-SDC/AI-SDC/pull/165))
*    Fix OS system calls relying on calling "python" ([#162](https://github.com/AI-SDC/AI-SDC/pull/162))
*    Fix invalid command line argument in worst case attack example ([#164](https://github.com/AI-SDC/AI-SDC/pull/164))
*    Add current output JSON format documentation ([#168](https://github.com/AI-SDC/AI-SDC/pull/168))
*    Add current attack config format documentation ([#168](https://github.com/AI-SDC/AI-SDC/pull/168))

## Version 1.0.4 (May 5, 2023)

Changes:
*    Fixed SafeRandomForestClassifier "base estimator changed" error ([#143](https://github.com/AI-SDC/AI-SDC/pull/143))

## Version 1.0.3 (May 2, 2023)

Changes:
*    Refactored metrics ([#111](https://github.com/AI-SDC/AI-SDC/pull/111))
*    Fixed a bug making a report when dummy reps is 0 ([#113](https://github.com/AI-SDC/AI-SDC/pull/113))
*    Fixed safemodel JSON output ([#115](https://github.com/AI-SDC/AI-SDC/pull/115))
*    Added a module to produce recommendations from attack JSON output ([#116](https://github.com/AI-SDC/AI-SDC/pull/116))
*    Disabled non-default report logs ([#123](https://github.com/AI-SDC/AI-SDC/pull/123))
*    Fixed a minor bug in worst case example ([#124](https://github.com/AI-SDC/AI-SDC/pull/124))

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
