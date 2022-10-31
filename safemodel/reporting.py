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

    inter_params = {
        "attr": None,
        "batch_size": None,
        "current_epsilon": None,
        "cur_val": None,
        "diffs_list": "",
        "e": None,
        "epochs": None,
        "er": None,
        "error": None,
        "idx": None,
        "item": None,
        "key": None,
        "layer": None,
        "length": None,
        "match": "",
        "name": None,
        "num1": None,
        "num2": None,
        "num_samples": None,
        "model_type": None,
        "msg": None,
        "operator": None,
        "optimizer": None,
        "opt_msg": None,
        "suffix": None,
        "v1": None,
        "v2": None,
        "val": None,
    }
    # initialise all the values from the keyword parameters
    # check for parameters in keyword arguments.
    # if keyword is found set dictionary item value to that
    # provided by the keyword
    # if keyword is not found dictionary item value remains None,
    # and an error message is returned.

    for param, value in the_kwargs.items():
        inter_params[param] = value

    if inter_params["name"] is None:
        return "Error - get_reporting_string: No 'name' given"

    # A dictionary of f-strings follows

    report_string = {
        "NULL": (""),
        "less_than_min_value": (
            f"- parameter {inter_params['key']} = {inter_params['cur_val']}"
            f" identified as less than the recommended min value of {inter_params['val']}."
        ),
        "greater_than_max_value": (
            f"- parameter {inter_params['key']} = {inter_params['cur_val']}"
            f" identified as greater than the recommended max value of {inter_params['val']}."
        ),
        "different_than_fixed_value": (
            f"- parameter {inter_params['key']} = {inter_params['cur_val']}"
            f" identified as different than the recommended fixed value of {inter_params['val']}."
        ),
        "different_than_recommended_type": (
            f"- parameter {inter_params['key']} = {inter_params['cur_val']}"
            f" identified as different type to recommendation of {inter_params['val']}.\n"
        ),
        "change_param_type": (
            f"\nChanged parameter type for {inter_params['key']} to {inter_params['val']}.\n"
        ),
        "not_implemented_for_change": (
            f"Nothing currently implemented to change type of parameter {inter_params['key']} "
            f"from {type(inter_params['cur_val']).__name__} to {inter_params['val']}.\n"
        ),
        "changed_param_equal": (
            f"\nChanged parameter {inter_params['key']} = {inter_params['val']}.\n"
        ),
        "unknown_operator": (
            f"- unknown operator in parameter specification {inter_params['operator']}"
        ),
        "warn_possible_disclosure_risk": (
            "WARNING: model parameters may present a disclosure risk:\n"
        ),
        "within_recommended_ranges": (
            "Model parameters are within recommended ranges.\n"
        ),
        "error_not_called_fit": (
            "Error: user has not called fit() method or has deleted saved values."
        ),
        "recommend_do_not_release": ("Recommendation: Do not release."),
        "layer_configs_differ": (
            f"Layer {inter_params['layer']} configs "
            f"differ in {inter_params['length']} places:\n"
        ),
        "error_reloading_model_v1": (
            f"Error re-loading  model from {inter_params['v1']}:  {inter_params['e']}"
        ),
        "error_reloading_model_v2": (
            f"Error re-loading  model from {inter_params['v2']}: {inter_params['e']}"
        ),
        "same_ann_config": ("configurations match"),
        "different_layer_count": ("models have different numbers of layers"),
        "batch_size_zero": (
            "Batch size of 0 not allowed"
            "setting batch_size =32.\n"
            "Alter self.batch_size manually if required."
        ),
        "division_by_zero": ("Division by zero setting batch_size =1"),
        "dp_requirements_met": (
            "The requirements for DP are met, "
            f"current epsilon is: {inter_params['current_epsilon']}."
            "Calculated from the parameters:  "
            f"Num Samples = {inter_params['num_samples']}, "
            f"batch_size = {inter_params['batch_size']}, epochs = {inter_params['epochs']}.\n"
        ),
        "dp_requirements_not_met": (
            f"The requirements for DP are not met, "
            f"current epsilon is: {inter_params['current_epsilon']}.\n"
            f"To attain recommended DP the following parameters can be changed:  "
            f"Num Samples = {inter_params['num_samples']},"
            f"batch_size = {inter_params['batch_size']},"
            f"epochs = {inter_params['epochs']}.\n"
        ),
        "basic_params_differ": (
            "Warning: basic parameters differ in " f"{inter_params['length']} places:\n"
        ),
        "param_changed_from_to": (
            f"parameter {inter_params['key']} changed from {inter_params['val']} "
            f"to {inter_params['cur_val']} after model was fitted.\n"
        ),
        "unable_to_check": (
            f"Unable to check as an exception occurred: {inter_params['error']}"
        ),
        "neither_tree_trained": ("neither tree trained"),
        "tree1_not_trained": ("tree1 not trained"),
        "tree2_not_trained": ("tree2 not trained"),
        "internal_attribute_differs": (
            f"internal tree attribute {inter_params['attr']} differs\n"
        ),
        "exception_occurred": (f"An exception occurred: {inter_params['error']}"),
        "unexpected_item": (
            "unexpected item in curr_seperate dict "
            " passed by generic additional checks."
        ),
        "warn_fitted_different_base": (
            "Warning: model was fitted with different base estimator type.\n"
        ),
        "error_model_not_fitted": ("Error: model has not been fitted to data.\n"),
        "current_item_removed": (
            f"Error, item  {inter_params['item']} "
            "present in  saved but not current model.\n"
        ),
        "saved_item_removed": (
            f"Error, item  {inter_params['item']} "
            "present in  current but not saved model.\n"
        ),
        "both_item_removed": f"Note that item {inter_params['item']} missing from both versions.\n",
        "trees_edited": (
            "Error: current version of model has had trees manually edited.\n"
        ),
        "different_num_estimators": (
            f"Fitted model has {inter_params['num2']} estimators "
            f"but requested version has {inter_params['num1']}.\n"
        ),
        "forest_estimators_differ": (
            f"{inter_params['idx']} forest base estimators have been changed.\n"
        ),
        "unable_to_check_item": (
            "In SafeRandomForest.additional_checks: "
            f"Unable to check {inter_params['item']} as an exception occurred:"
            f"{inter_params['error']}.\n"
        ),
        "structure_differences": (
            f"structure {inter_params['item']} has "
            f"{len(inter_params['diffs_list'])} differences: "
            f"{inter_params['diffs_list']}"
        ),
        "no_dp_gradients_key": (
            "optimizer does not contain key _was_dp_gradients_called" " so is not DP."
        ),
        "found_dp_gradients_key": (
            "optimizer does  contain key _was_dp_gradients_called" " so should be DP."
        ),
        "changed_opt_no_fit": (
            "optimizer has been changed but fit() has not been rerun."
        ),
        "dp_optimizer_run": (
            " value of optimizer._was_dp_gradients_called is True, "
            "so DP variant of optimizer has been run"
        ),
        "unrecognised_combin ation": ("unrecognised combination"),
        "optimizer_allowed": (f"optimizer {inter_params['optimizer']} allowed"),
        "optimizer_not_allowed": (
            f"optimizer {inter_params['optimizer']} is not allowed"
        ),
        "using_dp_sgd": ("Changed parameter optimizer = 'DPKerasSGDOptimizer'"),
        "using_dp_adagrad": ("Changed parameter optimizer = 'DPKerasAdagradOptimizer'"),
        "using_dp_adam": ("Changed parameter optimizer = 'DPKerasAdamOptimizer'"),
        "during_compilation": (f"During compilation: {inter_params['opt_msg']}"),
        "recommend_not_release": (
            f"Recommendation is not to release because {inter_params['msg']}.\n"
        ),
        "error_saving_file": (
            f"saving as a {inter_params['suffix']} "
            f"file gave this error message:  {inter_params['er']}"
        ),
        "loading_from_unsupported": (
            f"loading from a {inter_params['suffix']} "
            "file is currently not supported"
        ),
        "opt_config_changed": ("Optimizer config has been changed since training."),
        "epsilon_above_normal": (
            f"WARNING: epsilon {inter_params['current_epsilon']} "
            "is above normal max recommended value.\n"
            "Discussion with researcher needed.\n"
        ),
        "recommend_further_discussion": (
            f"Recommendation is further discussion needed " f"{inter_params['msg']}.\n"
        ),
        "recommend_allow_release": ("Recommendation is to allow release.\n"),
        "allow_release_eps_below_max": (
            "Recommendation: Allow release.\n"
            f"Epsilon vale of model {inter_params['current_epsilon']} "
            "is below normal max recommended value.\n"
        ),
        "input_filename_with_extension": (
            "Please input a name with extension for the model to be saved."
        ),
        "filename_must_indicate_type": ("file name must indicate type as a suffix"),
        "suffix_not_supported_for_type": (
            f"{inter_params['suffix']} file suffix  not supported "
            f"for models of type {inter_params['model_type']}.\n"
        ),
    }

    # return the correct formatted string

    return report_string[inter_params["name"]]
