def get_reporting_string(**kwargs):

    the_kwargs = kwargs

    # initialise all the values 
    
    if "name" in kwargs.keys():
        name = the_kwargs["name"]
    else:
        return("No 'name' given")
        
    if "cur_val" in kwargs.keys():
        cur_val = the_kwargs["cur_val"]
    else:
        cur_val = 0
        
    if "val" in kwargs.keys():
        val = the_kwargs["val"]
    else:
        val = 0
        
    if "key" in kwargs.keys():
        key = the_kwargs["key"]
    else:
        key = 0
        
    if "operator" in kwargs.keys():
        operator = thekwargs["operator"]
    else:
        operator = ""

    if "layer" in kwargs.keys():
        layer = the_kwargs["layer"]
    else:
        layer = ""


    if "match" in kwargs.keys():
        match = the_kwargs["match"]
    else:
        match = ""

    if "v1" in kwargs.keys():
        v1 = the_kwargs["v1"]
    else:
        v1 = ""

    if "v2" in kwargs.keys():
        v2 = the_kwargs["v2"]
    else:
        v2 = ""
        
    if "e" in kwargs.keys():
        e = the_kwargs["e"]
    else:
        e = ""

    if "current_epsilon" in kwargs.keys():
        current_epsilon = the_kwargs["current_epsilon"]
    else:
        current_epsilon = 0
        
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
            f"Layer {layer} configs differ in {len(match)} places:\n"
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
        ),
        
        'dp_requirements_not_met': (
            f"The requirements for DP are not met, "
            f"current epsilon is: {current_epsilon}.\n"
        ),

        'basic_params_differ': (
            "Warning: basic parameters differ in {len(match)} places:\n"
        )
        
        
     
    }


    
    return (REPORT_STRING[name])
