### v.1.0.6 (Completion by July 3, 2023?) - Plan/Tasks
* Plan: Getting to `TARGET.json` :heavy_plus_sign: `ATTACK_CONFIG.json` :arrow_right: `ATTACK_RESULTS.json` :arrow_right: `SUMMARY.txt`
    - [] Improve `ATTACK_CONFIG.json`
        - [ ] Refactor attack argument handling: replace the attack args classes with a dictionary within each attack class and move config file handling into the `Attack` class [Issue [#156](https://github.com/AI-SDC/AI-SDC/issues/156)] [Assigned: SM - phase1]
        - [ ] Ensure this is documented and appears in the github.io docs [Assigned: SM - phase1]
        - [ ] Ensure that the config file is standardised to run any attack [Issue [#148](https://github.com/AI-SDC/AI-SDC/issues/148)] [Assigned: SM - phase1]
    * Create `TARGET.json` - *some of these must be done in sequence* [Assigned: RP - phase1]
        * Rename `dataset.py` to `target.py` and the `Data` class to `Target`; update all files that import it; extend the class to include the target model details and write the output a `TARGET.json` file - possibly add functions to automatically extract details from either a saved model or a `BaseEstimator` etc. Likely need to save the trained model and data to files and provide the path in the json.
        * Modify existing examples to add the target model information to the `Target` class.
        * Add a function to the `Target` class to enable a researcher to add information (str) - include safemodels.
        * Modify safemodels `request_release()` to add the target model information to the `Target` class and create the `TARGET.json` above. [Assigned: JS/RP]
        * Redirect safemodel info/outputs from `*checkfile.txt` to `TARGET.json`. Update the safemodel example(s) [Issue [#158](https://github.com/AI-SDC/AI-SDC/issues/158)] [Assigned: JS/RP - phase1]
        * Add generalisation error reporting to safemodels (and include in `TARGET.json`) [Issue [#150](https://github.com/AI-SDC/AI-SDC/issues/150)] [Assigned: JS/RP - phase1]
    * Create `ATTACK_RESULTS.json`
        * Redirect attack output to a single file instead of creating individual files for each attack [Assigned: YJ - phase1]
        * Add the ability to produce `ATTACK_RESULTS.json` from the command line with `TARGET.json` and `ATTACK_CONFIG.json` [Assigned: SM- phase2]
        * Add the user story where a path to predicted probabilities are included in `TARGET.json` instead of raw data to help support R etc. [Assigned: YJ- - phase3]
        * Refactor the attacks and ensure consistency of output format (including AIA) [Issue [#147](https://github.com/AI-SDC/AI-SDC/issues/147)] [Assigned: - phase3]
    * Improve `summary.txt`
        * Report generation should produce `summary.txt` by taking as input the `TARGET.json` and `ATTACK_RESULTS.json` files (*depends on completion of above*) [Issue [#152](https://github.com/AI-SDC/AI-SDC/issues/152)] [Assigned: YJ/SR -phase 2 ]
        * Refactor analysis modules to ensure code reuse [Issue [#127](https://github.com/AI-SDC/AI-SDC/issues/127)] [Assigned: YJ-phase3]
        * Improve the summary recommendations; how? is this sufficient/completed? (*needs discussion*) [Issue [#110](https://github.com/AI-SDC/AI-SDC/issues/110)] [Assigned: YJ-phase3]
* To go in v.1.0.6 or v.1.0.7?
    * Add `SafeXGBoostClassifier` and include rules data-mined by RP [Issue [#157](https://github.com/AI-SDC/AI-SDC/issues/157)] [Assigned: SR]
    * Add user stories to a documentation page (.rst) and link to examples/notebooks so users can find their specific use case? [Issue [#141](https://github.com/AI-SDC/AI-SDC/issues/141)]  [Assigned: ??]
    * Check for user adding attribute before calling fit() in safe models [Issue [#32](https://github.com/AI-SDC/AI-SDC/issues/32)] [Assigned: JS]
    * Improve tests
        * (*first needs breaking up into smaller PRs*) [Issue [#76](https://github.com/AI-SDC/AI-SDC/issues/76)] [Assigned: ?]
        * Output module output tests [Issue [#129](https://github.com/AI-SDC/AI-SDC/issues/129)] [Assigned: ?]
