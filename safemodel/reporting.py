"""methods to producte standard reporting strings"""
def get_reporting_string(**kwargs):
    """returns a standard formatted string from a diction of f-strings.

    Parameters
    ----------

    name: string
         The dictionary key and the name of the string to return.
    all-the-keywords: Any Type
         one or more values expected in the f-string.

    Returns
    -------

    msg: string
         A standard message string.

    Notes
    -----

    Sometimes an f-string has no parameters.
    Sometimes there are multiple parameters embedded in the f-string.

    """
    the_kwargs = kwargs

    # check for the name used to reference the diction ary of f-strings
    # if no name is found return an error string

    if "name" in kwargs.keys():
        name = the_kwargs["name"]
    else:
        return "Error - get_reporting_string: No 'name' given"


    # initialise all the values from the keyword parameters
    #check for parameters in keyword arguments.
    #if keyword is found set variable to that provided by the keyword
    #if keyword is not found set variable to None

    if "cur_val" in kwargs.keys():
        cur_val = the_kwargs["cur_val"]
    else:
        cur_val = None

    #--------------------------------------------
    if "val" in kwargs.keys():
        val = the_kwargs["val"]
    else:
        val = None

    #--------------------------------------------
    if "key" in kwargs.keys():
        key = the_kwargs["key"]
    else:
        key = None
    #--------------------------------------------
    if "operator" in kwargs.keys():
        operator = the_kwargs["operator"]
    else:
        operator = None
    #--------------------------------------------
    if "layer" in kwargs.keys():
        layer = the_kwargs["layer"]
    else:
        layer = None
    #--------------------------------------------
    if "match" in kwargs.keys():
        match = the_kwargs["match"]
    else:
        match = ""
    #--------------------------------------------

    if "v1" in kwargs.keys():
        v1 = the_kwargs["v1"]
    else:
        v1 = None
    #--------------------------------------------
    if "v2" in kwargs.keys():
        v2 = the_kwargs["v2"]
    else:
        v2 = None
    #--------------------------------------------
    if "e" in kwargs.keys():
        e = the_kwargs["e"]
    else:
        e = None
    #--------------------------------------------
    if "current_epsilon" in kwargs.keys():
        current_epsilon = the_kwargs["current_epsilon"]
    else:
        current_epsilon = None
    #--------------------------------------------
    if "num_samples" in kwargs.keys():
        num_samples = the_kwargs["num_samples"]
    else:
        num_samples = None
    #--------------------------------------------
    if "batch_size" in kwargs.keys():
        batch_size = the_kwargs["batch_size"]
    else:
        batch_size = None
    #--------------------------------------------

    if "epochs" in kwargs.keys():
        epochs = the_kwargs["epochs"]
    else:
        epochs = None
    #--------------------------------------------
    if "error" in kwargs.keys():
        error = the_kwargs["error"]
    else:
        error = None
    #--------------------------------------------
    if "er" in kwargs.keys():
        er = the_kwargs["er"]
    else:
        er = None
    #--------------------------------------------

    if "attr" in kwargs.keys():
        attr = the_kwargs["attr"]
    else:
        attr = None
    #--------------------------------------------

    if "num1" in kwargs.keys():
        num1 = the_kwargs["num1"]
    else:
        num1 = None
    #--------------------------------------------
    if "num2" in kwargs.keys():
        num2 = the_kwargs["num2"]
    else:
        num2 = None
    #--------------------------------------------
    if "idx" in kwargs.keys():
        idx = the_kwargs["idx"]
    else:
        idx = None
    #--------------------------------------------

    if "diffs_list" in kwargs.keys():
        diffs_list = the_kwargs["diffs_list"]
    else:
        diffs_list = ""
    #--------------------------------------------
    if "item" in kwargs.keys():
        item = the_kwargs["item"]
    else:
        item = None
    #--------------------------------------------
    if "optimizer" in kwargs.keys():
        optimizer = the_kwargs["optimizer"]
    else:
        optimizer = None
    #--------------------------------------------
    if "opt_msg" in kwargs.keys():
        opt_msg = the_kwargs["opt_msg"]
    else:
        opt_msg = None

    #--------------------------------------------
    if "msg" in kwargs.keys():
        msg = the_kwargs["msg"]
    else:
        msg = None
    #--------------------------------------------
    if "model_type" in kwargs.keys():
        model_type = the_kwargs["model_type"]
    else:
        model_type = None
    #--------------------------------------------

    if "model_type" in kwargs.keys():
        model_type = the_kwargs["model_type"]
    else:
        model_type = None

    #--------------------------------------------
    if "suffix" in kwargs.keys():
        suffix = the_kwargs["suffix"]
    else:
        suffix = None
    #--------------------------------------------

    if "length" in kwargs.keys():
        length = the_kwargs["length"]
    else:
        length = None

    # A dictionary of f-strings follows

    REPORT_STRING = {
        'NULL': (
            ''
        ),

        'less_than_min_value' :(
            f"- parameter {key} = {cur_val}"
            f" identified as less than the recommended min value of {val}."
        ),

        'greater_than_max_value': (
            f"- parameter {key} = {cur_val}"
            f" identified as greater than the recommended max value of {val}."
        ),

        'different_than_fixed_value': (
            f"- parameter {key} = {cur_val}"
            f" identified as different than the recommended fixed value of {val}."
        ),

        'different_than_reccomended_type': (
            f"- parameter {key} = {cur_val}"
            f" identified as different type to recommendation of {val}."
        ),

        'change_param_type': (
            f"\nChanged parameter type for {key} to {val}.\n"
        ),

        'not_implemented_for_change': (
            f"Nothing currently implemented to change type of parameter {key} "
            f"from {type(cur_val).__name__} to {val}.\n"
        ),


        'changed_param_equal': f"\nChanged parameter {key} = {val}.\n",

        'unknown_operator' : (
            f"- unknown operator in parameter specification {operator}"
        ),

        'warn_possible_disclosure_risk': (
            "WARNING: model parameters may present a disclosure risk:\n"
        ),

        'within_recommended_ranges': (
            "Model parameters are within recommended ranges.\n"
        ),

        'error_not_called_fit':(
            "Error: user has not called fit() method or has deleted saved values."
        ),

        'recommend_do_not_release':(
            "Recommendation: Do not release."
        ),


        'layer_configs_differ': (
            f"Layer {layer} configs differ in {length} places:\n"
        ),

        'error_reloading_model_v1': (
            f"Error re-loading  model from {v1}:  {e}"

        ),


        'error_reloading_model_v2': (
            f"Error re-loading  model from {v2}: {e}"
        ),

        'division_by_zero': (
            "Division by zero setting batch_size =1"
        ),

        'dp_requirements_met' : (
            "The requirements for DP are met, "
            f"current epsilon is: {current_epsilon}."
            "Calculated from the parameters:  "
            f"Num Samples = {num_samples}, "
            f"batch_size = {batch_size}, epochs = {epochs}.\n"
        ),

        'dp_requirements_not_met': (
            f"The requirements for DP are not met, "
            f"current epsilon is: {current_epsilon}.\n"
            f"To attain recommended DP the following parameters can be changed:  "
            f"Num Samples = {num_samples},"
            f"batch_size = {batch_size},"
            f"epochs = {epochs}.\n"
        ),

        'basic_params_differ': (
            f"Warning: basic parameters differ in {length} places:\n"
        ),

        'unable_to_check' : (
            f"Unable to check as an exception occurred: {error}"
        ),


        'neither_tree_trained': (
            "neither tree trained"
        ),

        'tree1_not_trained': (
            "tree1 not trained"
        ),

        'tree2_not_trained': (
            "tree2 not trained"
        ),

        'internal_attribute_differs' : (
            f"internal tree attribute {attr} differs\n"
        ),

        'exception_occurred' : (
            f"An exception occurred: {error}"
        ),

        'unexpected_item': (
            "unexpected item in curr_seperate dict "
            " passed by generic additional checks."
        ),

        'warn_fitted_different_base' : (
            "Warning: model was fitted with different base estimator type.\n"
        ),

        'error_model_not_fitted' : (
            "Error: model has not been fitted to data.\n"
        ),

        'trees_removed': (
            "Error: current version of model has had trees removed after fitting.\n"
        ),

        'trees_edited' : (
            "Error: current version of model has had trees manually edited.\n"
        ),

        'different_num_estimators' : (
            f"Fitted model has {num2} estimators "
            f"but requested version has {num1}.\n"
        ),

        'forest_estimators_differ': (
            f"Forest base estimators {idx} differ."
        ),

        'unable_to_check_item' : (
            "In SafeRandomForest.additional_checks: "
            f"Unable to check {item} as an exception occurred: {error}.\n"
        ),

        'structure_differences': (
            f"structure {item} has {len(diffs_list)} differences: {diffs_list}"
        ),

        'no_dp_gradients_key' : (
            "optimizer does not contain key _was_dp_gradients_called"
            " so is not DP."
        ),

        'found_dp_gradients_key' : (
            "optimizer does  contain key _was_dp_gradients_called"
            " so should be DP."
        ),

        'changed_opt_no_fit' : (
            "optimizer has been changed but fit() has not been rerun."
        ),

        'dp_optimizer_run' :(
                " value of optimizer._was_dp_gradients_called is True, "
                "so DP variant of optimizer has been run"
        ),

        'unrecognised_combin ation' : (
            "unrecognised combination"
        ),

        'optimizer_allowed' : (
            f"optimizer {optimizer} allowed"
        ),

        'optimizer_not_allowed' : (
            f"optimizer {optimizer} is not allowed"
        ),

        'using_dp_sgd' : (
            "Changed parameter optimizer = 'DPKerasSGDOptimizer'"
        ),

        'using_dp_adagrad' : (
            "Changed parameter optimizer = 'DPKerasAdagradOptimizer'"
        ),

        'using_dp_adam' : (
            "Changed parameter optimizer = 'DPKerasAdamOptimizer'"
        ),

        'during_compilation' : (
            f"During compilation: {opt_msg}"
        ),

        'recommend_not_release' : (
            f"Recommendation is not to release because {msg}.\n"
        ),

        'error_saving_file' : (
            f"saving as a {suffix} file gave this error message:  {er}"
        ),

        'loading_from_unsupported': (
            f"loading from a {suffix} file is currently not supported"
        ),

        'opt_config_changed' : (
            "Optimizer config has been changed since training."
        ),

        'epsilon_above_normal' : (
            f"WARNING: epsilon {current_epsilon} "
            "is above normal max recommended value.\n"
            "Discussion with researcher needed.\n"
        ),

        'recommend_further_discussion' : (
            f"Recommendation is further discussion needed "
            f"{msg}.\n"
        ),

        'recommend_allow_release' : (
            "Recommendation is to allow release.\n"
        ),

        'allow_release_eps_below_max' : (
            "Recommendation: Allow release.\n"
            f"Epsilon vale of model {current_epsilon} "
            "is below normal max recommended value.\n"
        ),

        'input_filename_with_extension': (
            "Please input a name with extension for the model to be saved."
        ),

        'filename_must_indicate_type' : (
            "file name must indicate type as a suffix"
        ),

        'suffix_not_supported_for_type' : (
            f"{suffix} file suffix  not supported "
            f"for models of type {model_type}.\n"
        )



    }


    # return the correct formatted string


    return REPORT_STRING[name]
