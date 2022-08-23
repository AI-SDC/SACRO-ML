REPORT_STRING = {
    'NULL': (
        ''
    ),

    'less_than_min_value' :(
        f"- parameter {key} = {cur_val}"
        f" identified as less than the recommended min value of {val}."
    ),

    'more_than_max_value': (
        f"- parameter {key} = {cur_val}"
        f" identified as greater than the recommended max value of {val}."
    ),

    'differenet_than_fixed_value': (
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

    
    'changed_parameter_equal': f"\nChanged parameter {key} = {val}.\n",

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
    )
}
