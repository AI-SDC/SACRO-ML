"""This module contains prototypes of privacy safe model wrappers."""

from __future__ import annotations

import copy
import datetime
import getpass
import json
import logging
import pathlib
import pickle
from pickle import PicklingError
from typing import Any

import joblib
from dictdiffer import diff

from aisdc.attacks.attribute_attack import AttributeAttack
from aisdc.attacks.likelihood_attack import LIRAAttack
from aisdc.attacks.target import Target
from aisdc.attacks.worst_case_attack import WorstCaseAttack

# pylint : disable=too-many-branches
from .reporting import get_reporting_string

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


def check_min(key: str, val: Any, cur_val: Any) -> tuple[str, bool]:
    """Checks minimum value constraint.

    Parameters
    ----------

    key : string
         The dictionary key to examine.
    val : Any Type
         The expected value of the key.
    cur_val : Any Type
         The current value of the key.
    ..

    Returns
    -------

    msg : string
         A message string.
    disclosive : bool
         A boolean value indicating whether the model is potentially disclosive.

    Notes
    -----
    """
    if isinstance(cur_val, (int, float)):
        if cur_val < val:
            disclosive = True
            msg = get_reporting_string(
                name="less_than_min_value", key=key, cur_val=cur_val, val=val
            )
        else:
            disclosive = False
            msg = ""
        return msg, disclosive

    disclosive = True
    msg = get_reporting_string(
        name="different_than_recommended_type", key=key, cur_val=cur_val, val=val
    )
    return msg, disclosive


def check_max(key: str, val: Any, cur_val: Any) -> tuple[str, bool]:
    """Checks maximum value constraint.

    Parameters
    ----------

    key : string
         The dictionary key to examine.
    val : Any Type
         The expected value of the key.
    cur_val : Any Type
         The current value of the key.

    Returns
    -------

    msg : string
         A message string.
    disclosive : bool
         A boolean value indicating whether the model is potentially disclosive.

    Notes
    -----
    """
    if isinstance(cur_val, (int, float)):
        if cur_val > val:
            disclosive = True
            msg = get_reporting_string(
                name="greater_than_max_value", key=key, cur_val=cur_val, val=val
            )
        else:
            disclosive = False
            msg = ""
        return msg, disclosive

    disclosive = True
    msg = get_reporting_string(
        name="different_than_recommended_type", key=key, cur_val=cur_val, val=val
    )
    return msg, disclosive


def check_equal(key: str, val: Any, cur_val: Any) -> tuple[str, bool]:
    """Checks equality value constraint.

    Parameters
    ----------

    key : string
         The dictionary key to examine.
    val : Any Type
         The expected value of the key.
    cur_val : Any Type
         The current value of the key.

    Returns
    -------

    msg : string
         A message string.
    disclosive : bool
         A boolean value indicating whether the model is potentially disclosive.

    Notes
    -----
    """
    if cur_val != val:
        disclosive = True
        msg = get_reporting_string(
            name="different_than_fixed_value", key=key, cur_val=cur_val, val=val
        )
    else:
        disclosive = False
        msg = ""
    return msg, disclosive


def check_type(key: str, val: Any, cur_val: Any) -> tuple[str, bool]:
    """Checks the type of a value.

    Parameters
    ----------

    key : string
         The dictionary key to examine.
    val : Any Type
         The expected value of the key.
    cur_val : Any Type
         The current value of the key.

    Returns
    -------

    msg : string
         A message string.
    disclosive : bool
         A boolean value indicating whether the model is potentially disclosive.

    Notes
    -----
    """
    if type(cur_val).__name__ != val:
        disclosive = True
        msg = get_reporting_string(
            name="different_than_recommended_type", key=key, cur_val=cur_val, val=val
        )
    else:
        disclosive = False
        msg = ""
    return msg, disclosive


class SafeModel:  # pylint: disable = too-many-instance-attributes
    """Privacy protected model base class.

    Attributes
    ----------

    model_type : string
          A string describing the type of model. Default is "None".
    model:
          The Machine Learning Model.
    saved_model:
          A saved copy of the Machine Learning Model used for comparison.
    ignore_items : list
          A list of items to ignore when comparing the model with the
          saved_model.
    examine_separately_items : list
          A list of items to examine separately. These items are more
          complex datastructures that cannot be compared directly.
    filename : string
          A filename to save the model.
    researcher : string
          The researcher user-id used for logging

    Notes
    -----

    Examples
    --------
    >>> safeRFModel = SafeRandomForestClassifier()
    >>> safeRFModel.fit(X, y)
    >>> safeRFModel.save(name="safe.pkl")
    >>> safeRFModel.preliminary_check()
    >>> safeRFModel.request_release(path="safe", ext="pkl", target=target)
    WARNING: model parameters may present a disclosure risk:
    - parameter min_samples_leaf = 1 identified as less than the recommended min value of 5.
    Changed parameter min_samples_leaf = 5.

    Model parameters are within recommended ranges.
    """

    def __init__(self) -> None:
        """Super class constructor, gets researcher name."""
        self.model_type: str = "None"
        self.model = None
        self.saved_model = None
        self.model_load_file: str = "None"
        self.model_save_file: str = "None"
        self.ignore_items: list[str] = []
        self.examine_seperately_items: list[str] = []
        self.basemodel_paramnames = []
        self.filename: str = "None"
        self.researcher: str = "None"
        self.timestamp: str = "None"
        try:
            self.researcher = getpass.getuser()
        except (ImportError, KeyError, OSError):  # pragma: no cover
            self.researcher = "unknown"

    def get_params(self, deep=True):
        """Gets dictionary of parameter values
        restricted to those expected by base classifier.
        """
        the_params = {}
        for key, val in self.__dict__.items():
            if key in self.basemodel_paramnames:
                the_params[key] = val
        if deep:
            pass  # not implemented yet
        return the_params

    def save(self, name: str = "undefined") -> None:
        """Writes model to file in appropriate format.

        Note this is overloaded in SafeKerasClassifer
        to deal with tensorflow specifics.

        Parameters
        ----------

        name : string
             The name of the file to save

        Returns
        -------

        Notes
        -----

        No return value

        Optimizer is deliberately excluded.
        To prevent possible to restart training and thus
        possible back door into attacks.
        """

        self.model_save_file = name
        if self.model_save_file == "undefined":
            print("You must input a name with extension to save the model.")
        else:
            thename = self.model_save_file.split(".")
            # print(f'in save(), parsed filename is {thename}')
            if len(thename) == 1:
                print("file name must indicate type as a suffix")
            else:
                suffix = self.model_save_file.split(".")[-1]

                if (
                    suffix == "pkl" and self.model_type != "KerasModel"
                ):  # save to pickle
                    with open(self.model_save_file, "wb") as file:
                        try:
                            pickle.dump(self, file)
                        except (TypeError, AttributeError, PicklingError) as type_err:
                            print(
                                "saving a .pkl file is unsupported for model type:"
                                f"{self.model_type}."
                                f"Error message was {type_err}"
                            )

                elif (
                    suffix == "sav" and self.model_type != "KerasModel"
                ):  # save to joblib
                    try:
                        joblib.dump(self, self.model_save_file)
                    except (TypeError, AttributeError, PicklingError) as type_err:
                        print(
                            "saving as a .sav (joblib) file is not supported "
                            f"for models of type {self.model_type}."
                            f"Error message was {type_err}"
                        )
                #                  Overloaded in safekeras
                #                 elif suffix in ("h5", "tf") and self.model_type == "KerasModel":
                #                     try:
                #                         tf.keras.models.save_model(
                #                             self,
                #                             self.model_save_file,
                #                             include_optimizer=False,
                #                             # save_traces=False,
                #                             save_format=suffix,
                #                         )

                #                     except (ImportError, NotImplementedError) as exception_err:
                #                         print(
                #                             "saving as a {suffix} file gave this error message:"
                #                             f"{exception_err}"
                #                         )
                else:
                    print(
                        f"{suffix} file suffix currently not supported "
                        f"for models of type {self.model_type}.\n"
                    )

    ## Load functionality not needed
    # - provide directly by underlying pickle/joblib mechanisms
    # and safekeras provides its own to deal with tensorflow

    #     def load(self, name: str = "undefined") -> None:
    #         """reads model from file in appropriate format.
    #         Note that safekeras overloads this function.

    #         Optimizer is deliberately excluded in the save
    #         To prevent possible to restart training and thus
    #         possible back door into attacks.
    #         Thus optimizer cannot be loaded.
    #         """
    #         temp_file=None
    #         self.model_load_file = name
    #         if self.model_load_file == "undefined":
    #             print("You must input a file name with extension to load a model.")
    #         else:
    #             thename = self.model_save_file.split(".")
    #             suffix = self.model_save_file.split(".")[-1]

    #             if suffix == ".pkl":  # load from pickle
    #                 with open(self.model_load_file, "rb") as file:
    #                     temp_file = pickle.load(self, file)
    #             elif suffix == ".sav":  # load from joblib
    #                 temp_file = joblib.load(self, self.model_save_file)
    #             #safekeras overloads loads
    #             elif suffix in ("h5","tf")  and self.model_type != "KerasModel":
    #                 print("tensorflow objects saved as h5 or tf"
    #                       "can only be loaded into models of type SafeKerasClassifier"
    #                      )
    #             else:
    #                 print(f"loading from a {suffix} file is currently not supported")

    #         return temp_file

    def __get_constraints(self) -> dict:
        """Gets constraints relevant to the model type from the master read-only file."""
        rules: dict = {}
        rule_path = pathlib.Path(__file__).with_name("rules.json")
        with open(rule_path, encoding="utf-8") as json_file:
            parsed = json.load(json_file)
            rules = parsed[self.model_type]
        return rules["rules"]

    def __apply_constraints(
        self, operator: str, key: str, val: Any, cur_val: Any
    ) -> str:
        """Applies a safe rule for a given parameter."""
        if operator == "is_type":
            if (val == "int") and (type(cur_val).__name__ == "float"):
                self.__dict__[key] = int(self.__dict__[key])
                msg = get_reporting_string(name="change_param_type", key=key, val=val)
            elif (val == "float") and (type(cur_val).__name__ == "int"):
                self.__dict__[key] = float(self.__dict__[key])
                msg = get_reporting_string(name="change_param_type", key=key, val=val)
            else:
                msg = get_reporting_string(
                    name="not_implemented_for_change", key=key, cur_val=cur_val, val=val
                )
        else:
            setattr(self, key, val)
            msg = get_reporting_string(name="changed_param_equal", key=key, val=val)
        return msg

    def __check_model_param(
        self, rule: dict, apply_constraints: bool
    ) -> tuple[str, bool]:
        """Checks whether a current model parameter violates a safe rule.
        Optionally fixes violations.
        """
        disclosive: bool = False
        msg: str = ""
        operator: str = rule["operator"]
        key: str = rule["keyword"]
        val: Any = rule["value"]
        cur_val: Any = getattr(self, key)
        if operator == "min":
            msg, disclosive = check_min(key, val, cur_val)
        elif operator == "max":
            msg, disclosive = check_max(key, val, cur_val)
        elif operator == "equals":
            msg, disclosive = check_equal(key, val, cur_val)
        elif operator == "is_type":
            msg, disclosive = check_type(key, val, cur_val)
        else:
            msg = get_reporting_string(
                name="unknown_operator", key=key, val=val, cur_val=cur_val
            )
        if apply_constraints and disclosive:
            msg += self.__apply_constraints(operator, key, val, cur_val)
        return msg, disclosive

    def __check_model_param_and(
        self, rule: dict, apply_constraints: bool
    ) -> tuple[str, bool]:
        """Checks whether current model parameters violate a logical AND rule.
        Optionally fixes violations.
        """
        disclosive: bool = False
        msg: str = ""
        for arg in rule["subexpr"]:
            temp_msg, temp_disc = self.__check_model_param(arg, apply_constraints)
            msg += temp_msg
            if temp_disc:
                disclosive = True
        return msg, disclosive

    def __check_model_param_or(self, rule: dict) -> tuple[str, bool]:
        """Checks whether current model parameters violate a logical OR rule."""
        disclosive: bool = True
        msg: str = ""
        for arg in rule["subexpr"]:
            temp_msg, temp_disc = self.__check_model_param(arg, False)
            msg += temp_msg
            if not temp_disc:
                disclosive = False
        return msg, disclosive

    def preliminary_check(
        self, verbose: bool = True, apply_constraints: bool = False
    ) -> tuple[str, bool]:
        """Checks whether current model parameters violate the safe rules.
        Optionally fixes violations.

        Parameters
        ----------

        verbose : bool
             A boolean value to determine increased output level.

        apply_constraints : bool
             A boolean to determine whether identified constraints are
             to be upheld and applied.

        Returns
        -------

        msg : string
           A message string
        disclosive : bool
           A boolean value indicating whether the model is potentially
           disclosive.

        Notes
        -----
        """
        disclosive: bool = False
        msg: str = ""
        notok_start = get_reporting_string(name="warn_possible_disclosure_risk")
        ok_start = get_reporting_string(name="within_recommended_ranges")
        rules: dict = self.__get_constraints()
        for rule in rules:
            operator = rule["operator"]
            if operator == "and":
                temp_msg, temp_disc = self.__check_model_param_and(
                    rule, apply_constraints
                )
            elif operator == "or":
                temp_msg, temp_disc = self.__check_model_param_or(rule)
            else:
                temp_msg, temp_disc = self.__check_model_param(rule, apply_constraints)
            msg += temp_msg
            if temp_disc:
                disclosive = True

        if disclosive:
            msg = notok_start + msg
        else:
            msg = ok_start + msg

        if verbose:
            print("Preliminary checks: " + msg)
        return msg, disclosive

    def get_current_and_saved_models(self) -> tuple[dict, dict]:
        """Makes a copy of self.__dict__
        and splits it into dicts for the current and saved versions.
        """
        current_model = {}

        attribute_names_as_list = copy.copy(list(self.__dict__.keys()))

        for key in attribute_names_as_list:
            if key not in self.ignore_items:
                # logger.debug(f'copying {key}')
                try:
                    value = self.__dict__[key]  # jim added
                    current_model[key] = copy.deepcopy(value)
                except (copy.Error, TypeError) as key_type:
                    logger.warning("%s cannot be copied", key)
                    logger.warning(
                        "...%s error; %s", str(type(key_type)), str(key_type)
                    )

        saved_model = current_model.pop("saved_model", "Absent")

        # return empty dict if necessary
        if (
            saved_model == "Absent"
            or saved_model is None
            or not isinstance(saved_model, dict)
        ):
            saved_model = {}
        else:
            # final check in case fit has been called twice
            _ = saved_model.pop("saved_model", "Absent")

        return current_model, saved_model

    def examine_seperate_items(
        self, curr_vals: dict, saved_vals: dict
    ) -> tuple[str, bool]:
        """Comparison of more complex structures
        in the super class we just check these model-specific items exist
        in both current and saved copies.
        """
        msg = ""
        disclosive = False

        for item in self.examine_seperately_items:
            if curr_vals[item] == "Absent" and saved_vals[item] == "Absent":
                disclosive = True
                msg += get_reporting_string(name="both_item_removed", item=item)

            if curr_vals[item] == "Absent" and not saved_vals[item] == "Absent":
                msg += get_reporting_string(name="current_item_removed", item=item)
                disclosive = True

            if saved_vals[item] == "Absent" and not curr_vals[item] == "Absent":
                disclosive = True
                msg += get_reporting_string(name="saved_item_removed", item=item)

        if not disclosive:  # ok, so can call mode-specific extra checks
            msg, disclosive = self.additional_checks(curr_vals, saved_vals)

        return msg, disclosive

    def posthoc_check(self) -> tuple[str, bool]:
        """Checks whether model has been interfered with since fit() was last run."""

        disclosive = False
        msg = ""

        current_model, saved_model = self.get_current_and_saved_models()
        if len(saved_model) == 0:
            msg = get_reporting_string(name="error_not_called_fit")
            msg += get_reporting_string(name="recommend_do_not_release")
            disclosive = True

        else:
            # remove things we don't care about
            for item in self.ignore_items:
                _ = current_model.pop(item, "Absent")
                _ = saved_model.pop(item, "Absent")

            # break out things that need to be handled/examined in more depth
            curr_separate = {}
            saved_separate = {}
            for item in self.examine_seperately_items:
                curr_separate[item] = current_model.pop(item, "Absent")
                saved_separate[item] = saved_model.pop(item, "Absent")

            # comparison on list of "simple" parameters
            match = list(diff(current_model, saved_model, expand=True))
            num_differences = len(match)
            if num_differences > 0:
                disclosive = True
                msg += get_reporting_string(
                    name="basic_params_differ", length=num_differences
                )
                for this_match in match:
                    if this_match[0] == "change":
                        msg += get_reporting_string(
                            name="param_changed_from_to",
                            key=this_match[1],
                            val=this_match[2][1],
                            cur_val=this_match[2][0],
                        )
                    else:
                        msg += f"{this_match}"

            # comparison on model-specific attributes
            extra_msg, extra_disclosive = self.examine_seperate_items(
                curr_separate, saved_separate
            )
            msg += extra_msg
            if extra_disclosive:
                disclosive = True

        return msg, disclosive

    def additional_checks(
        self, curr_separate: dict, saved_separate: dict
    ) -> tuple[str, bool]:
        """Placeholder function for additional posthoc checks e.g. keras this
        version just checks that any lists have the same contents.

        Parameters
        ----------

        curr_separate : python dictionary

        saved_separate : python dictionary

        Returns
        -------

        msg : string
        A message string
        disclosive : bool
        A boolean value to indicate whether the model is potentially disclosive.

        Notes
        -----

        posthoc checking makes sure that the two dicts have the same set of
        keys as defined in the list self.examine_separately
        """

        msg = ""
        disclosive = False
        for item in self.examine_seperately_items:
            if isinstance(curr_separate[item], list):
                if len(curr_separate[item]) != len(saved_separate[item]):
                    msg += (
                        f"Warning: different counts of values for parameter {item}.\n"
                    )
                    disclosive = True
                else:
                    for i in range(len(saved_separate[item])):
                        difference = list(
                            diff(curr_separate[item][i], saved_separate[item][i])
                        )
                        if len(difference) > 0:
                            msg += (
                                f"Warning: at least one non-matching value "
                                f"for parameter list {item}.\n"
                            )
                            disclosive = True
                            break

        return msg, disclosive

    def request_release(self, path: str, ext: str, target: Target = None) -> None:
        """Saves model to filename specified and creates a report for the TRE
        output checkers.

        Parameters
        ----------
        path : string
            Path to save the outputs.
        ext : str
            File extension defining the model saved format, e.g., "pkl" or "sav".
        target : attacks.target.Target
            Contains model and dataset information.

        Notes
        -----
        If target is not null, then worst case MIA and attribute inference
        attacks are called via run_attack.
        """
        # perform checks
        msg_prel, disclosive_prel = self.preliminary_check(verbose=False)
        msg_post, disclosive_post = self.posthoc_check()
        # prepare results
        output: dict = {
            "researcher": self.researcher,
            "model_type": self.model_type,
            "details": msg_prel,
        }
        if hasattr(self, "k_anonymity"):
            output["k_anonymity"] = str(self.k_anonymity)
        if not disclosive_prel and not disclosive_post:
            output["recommendation"] = "Proceed to next step of checking"
        else:
            output["recommendation"] = "Do not allow release"
            output["reason"] = msg_prel + msg_post
        # Run attacks programmatically if possible
        attack_results_filename = "attack_results"
        if target is not None:
            for attack_name in ["worst_case", "lira", "attribute"]:
                output[f"{attack_name}_results"] = self.run_attack(
                    target, attack_name, path, attack_results_filename
                )
        # add timestamp
        now = datetime.datetime.now()
        self.timestamp = str(now.strftime("%Y-%m-%d %H:%M:%S"))
        output["timestamp"] = self.timestamp
        data = [output]
        # save output
        if target is None:
            target = Target(model=self)
        target.add_safemodel_results(data)
        target.save(path, ext)

    def run_attack(
        self,
        target: Target = None,
        attack_name: str = None,
        output_dir: str = "RES",
        report_name: str = "undefined",
    ) -> dict:
        """Runs a specified attack on the trained model and saves a report to file.

        Parameters
        ----------
        target : Target
            The target in the form of a Target object.
        attack_name : str
            Name of the attack to run.
        output_dir : str
            Name of the directory to store .json and .pdf output reports
        report_name : str
            Name of a .json file to save report.

        Returns
        -------
        dict
            Metadata results.

        Notes
        -----
        Currently implement attack types are:
        Likelihood Ratio: lira
        Worst_Case Membership inference: worst_case
        Single Attribute Inference: attributes
        """
        if attack_name == "worst_case":
            attack_obj = WorstCaseAttack(
                n_reps=10,
                n_dummy_reps=1,
                p_thresh=0.05,
                training_preds_filename=None,
                test_preds_filename=None,
                test_prop=0.5,
                output_dir=output_dir,
                report_name=report_name,
            )
            attack_obj.attack(target)
            output = attack_obj.make_report()
            metadata = output["metadata"]
        elif attack_name == "lira":
            attack_obj = LIRAAttack(
                n_shadow_models=100,
                output_dir=output_dir,
                report_name=report_name,
            )
            attack_obj.attack(target)
            output = attack_obj.make_report()
            metadata = output["metadata"]
        elif attack_name == "attribute":
            attack_obj = AttributeAttack(
                output_dir=output_dir,
                report_name=report_name,
            )
            attack_obj.attack(target)
            output = attack_obj.make_report()
            metadata = output["metadata"]
        else:
            metadata = {}
            metadata["outcome"] = "unrecognised attack type requested"
        print(f"attack {attack_name}, metadata {metadata}")
        return metadata

    def __str__(self) -> str:  # pragma: no cover
        """Returns string with model description.
        No point writing a test, especially as it depends on username.
        """
        return self.model_type + " with parameters: " + str(self.__dict__)
