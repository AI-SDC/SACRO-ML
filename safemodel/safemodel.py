"""This module contains prototypes of privacy safe model wrappers."""

from __future__ import annotations

import copy
import getpass
import json
import logging
import pathlib
import pickle
from typing import Any
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
import os

import joblib
from dictdiffer import diff

from attacks import worst_case_attack, dataset
from attacks.likelihood_attack import LIRAAttackArgs, LIRAAttack # pylint: disable = import-error


logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)




def check_min(key: str, val: Any, cur_val: Any) -> tuple[str, bool]:
    """Checks minimum value constraint.

    Parameters
    ----------

    key: string
         The dictionary key to examine.
    val: Any Type
         The expected value of the key.
    cur_val: Any Type
         The current value of the key.
    ..

    Returns
    -------

    msg: string
         A message string.
    disclosive: bool
         A boolean value indicating whether the model is potentially disclosive.

    Notes
    -----


    """
    if cur_val < val:
        disclosive = True
        msg = (
            f"- parameter {key} = {cur_val}"
            f" identified as less than the recommended min value of {val}."
        )
    else:
        disclosive = False
        msg = ""
    return msg, disclosive


def check_max(key: str, val: Any, cur_val: Any) -> tuple[str, bool]:
    """Checks maximum value constraint.

    Parameters
    ----------

    key: string
         The dictionary key to examine.
    val: Any Type
         The expected value of the key.
    cur_val: Any Type
         The current value of the key.

    Returns
    -------

    msg: string
         A message string.
    disclosive: bool
         A boolean value indicating whether the model is potentially disclosive.


    Notes
    -----


    """
    if cur_val > val:
        disclosive = True
        msg = (
            f"- parameter {key} = {cur_val}"
            f" identified as greater than the recommended max value of {val}."
        )
    else:
        disclosive = False
        msg = ""
    return msg, disclosive


def check_equal(key: str, val: Any, cur_val: Any) -> tuple[str, bool]:
    """Checks equality value constraint.



    Parameters
    ----------

    key: string
         The dictionary key to examine.
    val: Any Type
         The expected value of the key.
    cur_val: Any Type
         The current value of the key.

    Returns
    -------

    msg: string
         A message string.
    disclosive: bool
         A boolean value indicating whether the model is potentially disclosive.


    Notes
    -----


    """
    if cur_val != val:
        disclosive = True
        msg = (
            f"- parameter {key} = {cur_val}"
            f" identified as different than the recommended fixed value of {val}."
        )
    else:
        disclosive = False
        msg = ""
    return msg, disclosive


def check_type(key: str, val: Any, cur_val: Any) -> tuple[str, bool]:
    """Checks the type of a value.

    Parameters
    ----------

    key: string
         The dictionary key to examine.
    val: Any Type
         The expected value of the key.
    cur_val: Any Type
         The current value of the key.

    Returns
    -------

    msg: string
         A message string.
    disclosive: bool
         A boolean value indicating whether the model is potentially disclosive.

    Notes
    -----


    """
    if type(cur_val).__name__ != val:
        disclosive = True
        msg = (
            f"- parameter {key} = {cur_val}"
            f" identified as different type to recommendation of {val}."
        )
    else:
        disclosive = False
        msg = ""
    return msg, disclosive


class SafeModel: # pylint: disable = too-many-instance-attributes
    """Privacy protected model base class.

    Attributes
    ----------

    model_type: string
          A string describing the type of model. Default is "None".
    model:
          The Machine Learning Model.
    saved_model:
          A saved copy of the Machine Learning Model used for comparison.
    ignore_items: list
          A list of items to ignore when comparing the model with the
          saved_model.
    examine_separately_items: list
          A list of items to examine separately. These items are more
          complex datastructures that cannot be compared directly.
    filename: string
          A filename to save the model.
    researcher: string
          The researcher user-id used for logging



    Notes
    -----

    Examples
    --------
    >>> safeRFModel = SafeRandomForestClassifier()
    >>> safeRFModel.fit(X, y)
    >>> safeRFModel.save(name="safe.pkl")
    >>> safeRFModel.preliminary_check()
    >>> safeRFModel.request_release(filename="safe.pkl")
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
        self.filename: str = "None"
        self.researcher: str = "None"
        try:
            self.researcher = getpass.getuser()
        except (ImportError, KeyError, OSError):
            self.researcher = "unknown"

    def save(self, name: str = "undefined") -> None:
        """Writes model to file in appropriate format.

        Parameters
        ----------

        name: string
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
        while self.model_save_file == "undefined":
            self.model_save_file = input(
                "Please input a name with extension for the model to be saved."
            )

        thename = self.model_save_file.split(".")
        # print(f'in save(), parsed filename is {thename}')
        if len(thename) == 1:
            print("file name must indicate type as a suffix")
        else:
            suffix = self.model_save_file.split(".")[-1]

            if suffix == "pkl" and self.model_type != "KerasModel":  # save to pickle
                with open(self.model_save_file, "wb") as file:
                    try:
                        pickle.dump(self, file)
                    except TypeError as type_err:
                        print(
                            f"saving a .pkl file is unsupported for model type: {self.model_type}."
                            f"Error message was {type_err}"
                        )

            elif suffix == "sav" and self.model_type != "KerasModel":  # save to joblib
                try:
                    joblib.dump(self, self.model_save_file)
                except TypeError as type_err:
                    print(
                        "saving as a .sav (joblib) file is not supported "
                        f"for models of type {self.model_type}."
                        f"Error message was {type_err}"
                    )
            elif suffix in ("h5", "tf") and self.model_type == "KerasModel":
                try:
                    tf.keras.models.save_model(
                        self,
                        self.model_save_file,
                        include_optimizer=False,
                        # save_traces=False,
                        save_format=suffix,
                    )

                except tf.keras.NotImplementedError as exception_err:
                    print(f"saving as a {suffix} file gave this error message:  {exception_err}")
            else:
                print(
                    f"{suffix} file suffix currently not supported "
                    f"for models of type {self.model_type}.\n"
                )

    def load(self, name: str = "undefined") -> None:
        """reads model from file in appropriate format.
        Optimizer is deliberately excluded in the save
        To prevent possible to restart training and thus
        possible back door into attacks.
        Thus optimizer cannot be loaded.
        """

        self.model_load_file = name
        while self.model_load_file == "undefined":
            self.model_save_file = input(
                "Please input a name with extension for the model to load."
            )
        if self.model_load_file[-4:] == ".pkl":  # load from pickle
            with open(self.model_load_file, "rb") as file:
                temp_file = pickle.load(self, file)
        elif self.model_load_file[-4:] == ".sav":  # load from joblib
            temp_file = joblib.load(self, self.model_save_file)
        elif self.model_load_file[-3:] == ".h5":
            # load from .h5
            temp_file = tf.keras.models.load_model(
                self.model_load_file, custom_objects={"Safe_KerasModel": self}
            )

        elif self.model_load_file[-3:] == ".tf":
            # load from tf
            temp_file = tf.keras.models.load_model(
                self.model_load_file, custom_objects={"Safe_KerasModel": self}
            )

        else:
            suffix = self.model_load_file.split(".")[-1]
            print(f"loading from a {suffix} file is currently not supported")

        return temp_file

    def __get_constraints(self) -> dict:
        """Gets constraints relevant to the model type from the master read-only file."""
        rules: dict = {}
        rule_path = pathlib.Path(__file__).with_name("rules.json")
        with open(rule_path, "r", encoding="utf-8") as json_file:
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
                msg = f"\nChanged parameter type for {key} to {val}.\n"
            elif (val == "float") and (type(cur_val).__name__ == "int"):
                self.__dict__[key] = float(self.__dict__[key])
                msg = f"\nChanged parameter type for {key} to {val}.\n"
            else:
                msg = (
                    f"Nothing currently implemented to change type of parameter {key} "
                    f"from {type(cur_val).__name__} to {val}.\n"
                )
        else:
            setattr(self, key, val)
            msg = f"\nChanged parameter {key} = {val}.\n"
        return msg

    def __check_model_param(
        self, rule: dict, apply_constraints: bool
    ) -> tuple[str, bool]:
        """Checks whether a current model parameter violates a safe rule.
        Optionally fixes violations."""
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
            msg = f"- unknown operator in parameter specification {operator}"
        if apply_constraints and disclosive:
            msg += self.__apply_constraints(operator, key, val, cur_val)
        return msg, disclosive

    def __check_model_param_and(
        self, rule: dict, apply_constraints: bool
    ) -> tuple[str, bool]:
        """Checks whether current model parameters violate a logical AND rule.
        Optionally fixes violations."""
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

        verbose: bool
             A boolean value to determine increased output level.

        apply_constraints: bool
             A boolean to determine whether identified constraints are
             to be upheld and applied.

        Returns
        -------

        msg: string
           A message string
        disclosive: bool
           A boolean value indicating whether the model is potentially
           disclosive.


        Notes
        -----


        """
        disclosive: bool = False
        msg: str = ""
        rules: dict = self.__get_constraints()
        for rule in rules:
            operator = rule["operator"]
            if operator == "and":
                temp_msg, temp_disc = self.__check_model_param_and(rule, apply_constraints)
            elif operator == "or":
                temp_msg, temp_disc = self.__check_model_param_or(rule)
            else:
                temp_msg, temp_disc = self.__check_model_param(rule, apply_constraints)
            msg += temp_msg
            if temp_disc:
                disclosive = True
        if disclosive:
            msg = "WARNING: model parameters may present a disclosure risk:\n" + msg
        else:
            msg = "Model parameters are within recommended ranges.\n" + msg
        if verbose:
            print(msg)
        return msg, disclosive

    def get_current_and_saved_models(self) -> tuple[dict, dict]:
        """Makes a copy of self.__dict__
        and splits it into dicts for the current and saved versions
        """
        current_model = {}

        attribute_names_as_list = copy.copy(list(self.__dict__.keys()))

        for key in attribute_names_as_list:

            if key not in self.ignore_items:
                # logger.debug(f'copying {key}')
                try:
                    value = self.__dict__[key]  # jim added
                    current_model[key] = copy.deepcopy(value)
                except copy.Error as key_type:
                    logger.warning("%s cannot be copied", key)
                    logger.warning("...%s error; %s", str(type(key_type)), str(key_type))
            # logger.debug('...done')
        # logger.info('copied')

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

            # rename keys to get rid of the "a_" suffix
        #             keyscopy= copy(saved_model.keys())
        #             for oldkey in keyscopy:
        #                 newkey=oldkey[2:]
        #                 print(f' {oldkey} -> {newkey}')
        #                 saved_model[newkey]= saved_model.pop(oldkey)
        return current_model, saved_model

    def examine_seperate_items(
        self, curr_vals: dict, saved_vals: dict
    ) -> tuple[str, bool]:
        """comparison of more complex structures
        in the super class we just check these model-specific items exist
        in both current and saved copies"""
        msg = ""
        disclosive = False
        for item in self.examine_seperately_items:
            if curr_vals[item] == "Absent" and saved_vals[item] == "Absent":
                # not sure if this is necessarily disclosive
                msg += f"Note that item {item} missing from both versions"

            elif curr_vals[item] == "Absent" and not saved_vals[item] == "Absent":
                disclosive = True
                msg += f"Error, item {item} present in  saved but not current model"
            elif saved_vals[item] == "Absent" and not curr_vals[item] == "Absent":
                disclosive = True
                msg += f"Error, item {item} present in current but not saved model"
            else:  # ok, so can call mode-specific extra checks
                msg2, disclosive2 = self.additional_checks(curr_vals, saved_vals)
                if len(msg2) > 0:
                    msg += msg2
                if disclosive2:
                    disclosive = True
        return msg, disclosive

    def posthoc_check(self) -> tuple[str, bool]:
        """Checks whether model has been interfered with since fit() was last run"""

        disclosive = False
        msg = ""

        current_model, saved_model = self.get_current_and_saved_models()
        if len(saved_model) == 0:
            msg = "Error: user has not called fit() method or has deleted saved values."
            msg += "Recommendation: Do not release."
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
            if len(match) > 0:
                disclosive = True
                msg += f"Warning: basic parameters differ in {len(match)} places:\n"
                for this_match in match:
                    if this_match[0] == "change":
                        msg += f"parameter {this_match[1]} changed from {this_match[2][1]} "
                        msg += f"to {this_match[2][0]} after model was fitted.\n"
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
        version just checks that any lists have the same contents


        Parameters
        ----------

        curr_separate: python dictionary

        saved_separate: python dictionary


        Returns
        -------

        msg: string
        A message string
        disclosive: bool
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
                if saved_separate[item] == "Absent":
                    msg += f"Error: Saved copy is missing attribute {item}"
                    disclosive = True

                elif len(curr_separate[item]) != len(saved_separate[item]):
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

    def request_release(self, filename: str = "undefined",data_obj:dataset.Data=None) -> None:
        """Saves model to filename specified and creates a report for the TRE
        output checkers.

        Parameters
        ----------

        filename: string
        The filename used to save the model

        dataobj: object of type Data
        Contains train/test data and encoding dictionary needed to run attacks

        Returns
        -------


        Notes
        -----
         1. The dataset object is saved in a file called filebase_data.json
         (where filebase= filename without the extension)
         for reference/use by the TRE.
         Data should never be held or stored with the model.
         Clearly filebase_data.json mst never leave the TRE.
         2. If data_obj is not null, then worst case MIA and attribute inference
         attacks are called via run_attack.
         Outputs from the attacks will be stored in filebase_attack_res.json



        """
        if filename == "undefined":
            print("You must provide the name of the file you want to save your model")
            print("For security reasons, this will overwrite previous versions")
        else:
            self.save(filename)
            msg_prel, disclosive_prel = self.preliminary_check(verbose=False)
            msg_post, disclosive_post = self.posthoc_check()

            output: dict = {
                "researcher": self.researcher,
                "model_type": self.model_type,
                "model_save_file": self.model_save_file,
                "details": msg_prel,
            }
            if hasattr(self, "k_anonymity"):
                output["k_anonymity"] = f"{self.k_anonymity}"
            if not disclosive_prel and not disclosive_post:
                output[
                    "recommendation"
                ] = f"Run file {filename} through next step of checking procedure"
            else:
                output["recommendation"] = "Do not allow release"
                output["reason"] = msg_prel + msg_post

            ###needs testing between these triple comments
            if data_obj is not None:
                #make filenames and save a cpy of the data
                filebase= os.path.splitext(filename)[0]
                print(f'filebase is {filebase}')
                data_file = filebase +"_data.pickle"
                with open(data_file, 'wb') as handle:
                    pickle.dump(data_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

                for attack_name in ['worst_case','lira','attribute']:
                    print(f'running {attack_name} attack')
                    keyname=f"{attack_name}_results"
                    attackresfile= f"{filebase}_{attack_name}_res"
                    output[keyname]=  self.run_attack(data_obj,
                                                attack_name,
                                                attackresfile
                                               )
                    for key,val in output[keyname]:
                        print(f'{key}: {val}')
             ###

            json_str = json.dumps(output, indent=4)
            outputfilename = self.researcher + "_checkfile.json"
            with open(outputfilename, "a", encoding="utf-8") as file:
                file.write(json_str)


    def run_attack(self,
                   data_obj:dataset.Data=None,
                   attack_name:str="worst_case",
                   filename:str="undefined")->dict:

        """Runs a specified attack on the trained model and saves a report to file

        Parameters
        ----------
        data_obj: Data
        the dataset in the form of a Data object

        attack_name: string

        filebasename: string
        Report will be saved to filebasename.json


        Returns
        -------
        dict of meta data results

        Notes
        -----
        Currently implement attack types are:
        Likelihood Ratio: lira
        Worst_Case Membership inference: worst_case
        Single Attribute Inference: attributes
        """
        if attack_name == "worst_case":
            attack_args = worst_case_attack.WorstCaseAttackArgs(n_reps=10,
                # number of baseline (dummy) experiments to do
                n_dummy_reps=1,
                # Threshold to determine significance of things
                p_thresh=0.05,
                # Filename arguments needed by the code, meaningless if run programmatically
                in_sample_filename=None,
                out_sample_filename=None,
                # Proportion of data to use as a test set for the attack model;
                test_prop=0.5,
                # Report name is None - don't make json or pdf files
                report_name=None
            )
            attack_obj = worst_case_attack.WorstCaseAttack(attack_args)
            attack_obj.attack(dataset=data_obj,target_model=self)
            output = attack_obj.make_report()
            metadata = output['metadata']
            #print(f'metadata is a {type(metadata)}\n with contents {metadata}')

        elif attack_name=="lira":
            args = LIRAAttackArgs(n_shadow_models=100, report_name="lira_example_report")
            attack_obj = LIRAAttack(args)
            attack_obj.attack(data_obj,self)
            output = attack_obj.make_report() # also makes .pdf and .json files
            metadata = output['metadata']

        elif attack_name=="attribute":
            metadata = {'outcomes':"attribute inference attack not implemented within safemodel yet"}

        else:
            metadata= {"outcome":"unrecognised attack type requested"}

        try:
            with open(f'{filename}.json', 'w',encoding='utf-8') as fp:
                json.dump(metadata, fp)
        except:
            print(f'couldnt serialise metadata {metadata} for attack {attack_name}')

        return metadata


    def __str__(self) -> str:
        """Returns string with model description."""
        return self.model_type + " with parameters: " + str(self.__dict__)
