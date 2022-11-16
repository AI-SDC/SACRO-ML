""" safekeras.py:
 Jim Smith, Andrew McCarty and Richard Preen
 UWE 2022
"""
# general imports


import os
import warnings

# import sys
from typing import Any, Tuple

import numpy as np

# tensorflow imports
import tensorflow as tf
import tensorflow_privacy as tfp
from dictdiffer import diff
from tensorflow.keras import Model as KerasModel  # pylint: disable = import-error
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

# safemodel superclass
from ..reporting import get_reporting_string
from ..safemodel import SafeModel

# suppress numerous deprecatino warnings
# shut tensorflow up
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# this is the current class that dpvarians of optimizers come from
# may change in later versions of tensorflow_privacy
DP_CLASS_STRING = (
    "tensorflow_privacy.privacy.optimizers.dp_optimizer_keras."
    "make_keras_optimizer_class.<locals>.DPOptimizerClass"
)

DP_CLASS_STRING2 = (
    "tensorflow_privacy.privacy.optimizers.dp_optimizer_keras."
    "make_keras_generic_optimizer_class.<locals>.DPOptimizerClass"
)


def same_configs(m1: Any, m2: Any) -> Tuple[bool, str]:
    """Checks if two models havethe same architecture"""
    num_layers = len(m1.layers)
    if len(m2.layers) != num_layers:
        errstr = get_reporting_string(name="different_layer_count")
        return False, errstr
    for layer in range(num_layers):
        m1_layer_config = m1.layers[layer].get_config()
        _ = m1_layer_config.pop("name")
        m2_layer_config = m2.layers[layer].get_config()
        _ = m2_layer_config.pop("name")
        match = list(diff(m1_layer_config, m2_layer_config, expand=True))
        num_diffs = len(match)
        if num_diffs > 0:
            msg = get_reporting_string(
                name="layer_configs_differ", layer=layer, length=num_diffs
            )
            # f"Layer {layer} configs differ in {len(match)} places:\n"
            for i in range(num_diffs):
                if match[i][0] == "change":
                    msg += get_reporting_string(
                        name="param_changed_from_to",
                        key=match[i][1],
                        val=match[i][2][0],
                        cur_val=match[i][2][1],
                    )
                else:  # should not be reachable as dense objects cannot be modified
                    msg += f"{match[i]}"  # pragma: no cover
            return False, msg

    return True, get_reporting_string(name="same_ann_config")


def same_weights(m1: Any, m2: Any) -> Tuple[bool, str]:
    """checks if two nets with same architecture havethe same weights"""
    num_layers = len(m1.layers)
    if num_layers != len(m2.layers):
        return False, "different numbers of layers"
    # layer 0 is input layer determined by data size
    for layer in range(1, num_layers):
        m1layer = m1.layers[layer].get_weights()
        m2layer = m2.layers[layer].get_weights()
        if len(m1layer[0][0]) != len(m2layer[0][0]):
            return False, f"layer {layer} not the same size."
        for dim in range(len(m1layer)):  # pylint: disable=consider-using-enumerate
            m1d = m1layer[dim]
            m2d = m2layer[dim]
            # print(type(m1d), m1d.shape)
            if not np.array_equal(m1d, m2d):  # pragma: no cover
                return False, f"dimension {dim} of layer {layer} differs"
    return True, "weights match"


def check_checkpoint_equality(v1: str, v2: str) -> Tuple[bool, str]:
    """compares two checkpoints saved with tensorflow save_model
    On the assumption that the optimiser is not going to be saved,
    and that the model is going to be saved in frozen form
    this only checks the architecture and weights layer by layer
    """
    msg = ""
    same = True

    try:
        model1 = tf.keras.models.load_model(v1)
    except Exception as e:  # pylint:disable=broad-except
        msg = get_reporting_string(name="error_reloading_model_v1", e=e)
        # f"Error re-loading  model from {v1}:  {e}"
        return False, msg
    try:
        model2 = tf.keras.models.load_model(v2)
    except Exception as e:  # pylint:disable=broad-except
        msg = get_reporting_string(name="error_reloading_model_v2", e=e)
        # f"Error re-loading  model from {v2}: {e}"
        return False, msg

    same_config, config_message = same_configs(model1, model2)
    if not same_config:
        print("different config")
        msg += config_message
        same = False

    same_weight, weights_message = same_weights(model1, model2)
    if not same_weight:
        print("different weights")
        msg += weights_message
        same = False

    return same, msg


def check_DP_used(optimizer) -> Tuple[bool, str]:
    """checks whether the DP optimizer was actually the one used"""

    key_needed = "_was_dp_gradients_called"
    critical_val = optimizer.__dict__.get(key_needed, "missing")

    if critical_val is True:
        reason = get_reporting_string(name="dp_optimizer_run")
        DPused = True
    elif critical_val == "missing":
        reason = get_reporting_string(name="no_dp_gradients_key")
        DPused = False
    elif critical_val is False:
        reason = get_reporting_string(name="changed_opt_no_fit")
        DPused = False
    else:  # pragma: no cover
        # not currently reachable because optimizer class does
        # not support assignment
        # but leave in to future-proof
        reason = get_reporting_string(
            name="unrecognised_combination"
        )  # pragma: no cover
        DPused = False  # pragma: no cover

    return DPused, reason


def check_optimizer_allowed(optimizer) -> Tuple[bool, str]:
    """checks if the model's optimizer is in our white-list
    default setting is not allowed
    """
    allowed = False
    opt_type = str(type(optimizer))
    reason = get_reporting_string(name="optimizer_not_allowed", optimizer=opt_type)
    if (DP_CLASS_STRING in opt_type) or (DP_CLASS_STRING2 in opt_type):
        allowed = True
        reason = get_reporting_string(name="optimizer_allowed", optimizer=opt_type)

    return allowed, reason


def check_optimizer_is_DP(optimizer) -> Tuple[bool, str]:
    """checks whether optimizer is one of tensorflow's DP versions"""
    DPused = False
    reason = "None"
    if "_was_dp_gradients_called" not in optimizer.__dict__:
        reason = get_reporting_string(name="no_dp_gradients_key")
    else:
        reason = get_reporting_string(name="found_dp_gradients_key")
        DPused = True
    return DPused, reason


def load_safe_keras_model(name: str = "undefined") -> Tuple[bool, Any]:
    """
    reads model from file in appropriate format.
    Optimizer is deliberately excluded in the save.
    This is to prevent possibility of restarting training,
    which could offer possible back door into attacks.
    Thus optimizer cannot be loaded.
    """
    the_model = None
    model_load_file = name
    msg = ""
    if model_load_file == "undefined":
        msg = "Please input a name with extension for the model to load."

    elif model_load_file[-3:] == ".tf":
        # load from tf
        the_model = tf.keras.models.load_model(
            model_load_file  # , custom_objects={"SafeKerasModel"}
        )
        load = tf.keras.models.load_model(model_load_file, compile="False")
        the_model.set_weights(load.get_weights())

    else:
        suffix = model_load_file.split(".")[-1]
        msg = f"loading from a {suffix} file is currently not supported"

    if the_model is not None:
        return (True, the_model)
    # else
    return (False, msg)


class SafeKerasModel(KerasModel, SafeModel):
    """Privacy Protected Wrapper around  tf.keras.Model class from tensorflow 2.8
    disabling pylont warnings about number of instance attributes
    as this is necessarily complex
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Creates model and applies constraints to params"""

        # the_args = args
        the_kwargs = kwargs

        # initialise all the values that get provided as options to keras
        # and also l2 norm clipping and learning rates, batch sizes
        ##inputs = kwargs.get("inputs","notFound")
        ##if inputs=="notFound":
        ##    inputs = args[0] if len(args) == 3 else None
        if "inputs" in kwargs.keys():  # pylint: disable=consider-iterating-dictionary
            inputs = the_kwargs["inputs"]
        elif len(args) == 3:  # defaults is for Model(input,outputs,names)
            inputs = args[0]
        self.outputs = None
        ##outputs = kwargs.get("outputs","notFound")
        ##if outputs=="notFound":
        ##    outputs = args[1] if len(args) == 3 else None
        if "outputs" in kwargs.keys():  # pylint: disable=consider-iterating-dictionary
            outputs = the_kwargs["outputs"]
        elif len(args) == 3:
            # self.outputs = args[1]
            outputs = args[1]

        # call the keras super class first as this comes first in chain
        super().__init__(  # pylint: disable=unexpected-keyword-arg
            inputs=inputs, outputs=outputs
        )

        # set values where the user has supplied them
        # if not supplied set to a value that preliminary_check
        # will over-ride with TRE-specific values from rules.json
        defaults = {
            "l2_norm_clip": 1.0,
            "noise_multiplier": 0.5,
            "min_epsilon": 10,
            "delta": 1e-5,
            "batch_size": 25,
            "num_microbatches": None,
            "learning_rate": 0.1,
            "optimizer": tfp.DPKerasSGDOptimizer,
            "num_samples": 250,
            "epochs": 20,
            "current_epsilon": 999,
        }

        for key, val in defaults.items():
            if kwargs.get(key, "missing") != "missing":
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, val)

        if self.batch_size == 0:
            msg = get_reporting_string(name="batch_size_zero")
            print(msg)
            self.batch_size = 32

        SafeModel.__init__(self)

        self.model_type: str = "KerasModel"
        # remove. this from default class
        _ = self.__dict__.pop("saved_model")
        super().preliminary_check(apply_constraints=True, verbose=True)

    def dp_epsilon_met(
        self, num_examples: int, batch_size: int = 0, epochs: int = 0
    ) -> Tuple[bool, str]:
        """Checks if epsilon is sufficient for Differential Privacy
        Provides feedback to user if epsilon is not sufficient"""
        privacy = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
            n=num_examples,
            batch_size=batch_size,
            noise_multiplier=self.noise_multiplier,
            epochs=epochs,
            delta=self.delta,
        )
        ok = privacy[0] < self.min_epsilon
        return ok, privacy[0]

    def check_epsilon(
        self, num_samples: int, batch_size: int, epochs: int
    ) -> Tuple[bool, str]:
        """Computes the level of privacy guarantee is within recommended limits,
        and produces feedback"
        """
        msg = ""
        ok = False
        if batch_size == 0:
            msg += get_reporting_string(name="division_by_zero")
            batch_size = 1
        (
            ok,
            self.current_epsilon,  # pylint: disable=attribute-defined-outside-init
        ) = self.dp_epsilon_met(
            num_examples=num_samples, batch_size=batch_size, epochs=epochs
        )

        key_name = "dp_requirements_met" if ok else "dp_requirements_not_met"
        get_reporting_string(
            name=key_name,
            current_epsilon=self.current_epsilon,
            num_samples=num_samples,
            batch_size=batch_size,
            epochs=epochs,
        )
        print(msg)
        return ok, msg

    def compile(
        self, optimizer=None, loss="categorical_crossentropy", metrics=["accuracy"]
    ):  # pylint:disable=dangerous-default-value)
        """
        Replaces the optimiser with a DP variant if needed and creates the
        necessary DP params in the opt and loss dict, then calls tf compile.
        Allow None as default value for optimizer param because we explicitly
        deal with it.
        """
        replace_message = get_reporting_string(name="warn_possible_disclosure_risk")
        # "WARNING: model parameters may present a disclosure risk"
        using_DP_SGD = get_reporting_string(name="using_dp_sgd")
        # "Changed parameter optimizer = 'DPKerasSGDOptimizer'"
        Using_DP_Adagrad = get_reporting_string(name="using_dp_adagrad")
        # "Changed parameter optimizer = 'DPKerasAdagradOptimizer'"
        using_DP_Adam = get_reporting_string(name="using_dp_adam")
        # "Changed parameter optimizer = 'DPKerasAdamOptimizer'"

        optimizer_dict = {
            None: (using_DP_SGD, tfp.DPKerasSGDOptimizer),
            tfp.DPKerasSGDOptimizer: ("", tfp.DPKerasSGDOptimizer),
            tfp.DPKerasAdagradOptimizer: ("", tfp.DPKerasAdagradOptimizer),
            tfp.DPKerasAdamOptimizer: ("", tfp.DPKerasAdamOptimizer),
            "Adagrad": (
                replace_message + Using_DP_Adagrad,
                tfp.DPKerasAdagradOptimizer,
            ),
            "Adam": (replace_message + using_DP_Adam, tfp.DPKerasAdamOptimizer),
            "SGD": (replace_message + using_DP_SGD, tfp.DPKerasSGDOptimizer),
        }

        val = optimizer_dict.get(optimizer, "unknown")
        if val == "unknown":
            opt_msg = using_DP_SGD
            opt_used = tfp.DPKerasSGDOptimizer
        else:
            opt_msg = val[0]
            opt_used = val[1]

        self.optimizer = opt_used  # pylint: disable=attribute-defined-outside-init
        opt = opt_used(
            l2_norm_clip=self.l2_norm_clip,
            noise_multiplier=self.noise_multiplier,
            num_microbatches=self.num_microbatches,
            learning_rate=self.learning_rate,
        )

        if len(opt_msg) > 0:
            print(get_reporting_string(name="during_compilation", opt_msg=opt_msg))

        super().compile(opt, loss, metrics)

    def fit(  # pylint:disable=too-many-arguments
        self,
        X: Any,
        Y: Any,
        validation_data: Any,
        epochs: int,
        batch_size: int,
        refine_epsilon: bool = False,
    ) -> Any:
        """
        Overrides the tensorflow fit() method with some extra functionality:
        (i) records number of samples for checking DP epsilon values.
        (ii) does an automatic epsilon check and reports.
        (iia) if user sets refine_epsilon = true, return without fitting the model.
        (iii) then calls the tensorflow fit() function.
        (iv) finally makes a saved copy of the newly fitted model.
        """

        # pylint can't cope that we first declared these via a dict :(
        self.num_samples = X.shape[0]  # pylint: disable=attribute-defined-outside-init
        self.epochs = epochs  # pylint: disable=attribute-defined-outside-init
        self.batch_size = batch_size  # pylint: disable=attribute-defined-outside-init
        # make sure you are passing keywords through - but also checking batch size epochs
        ok, msg = self.check_epsilon(X.shape[0], batch_size, epochs)

        if not ok:
            print(msg)
        if refine_epsilon:
            print(
                "Not continuing with fitting model, "
                "as return epsilon was above max recommended value, "
                "and user set refine_epsilon= True"
            )
            return False, None

        returnval = super().fit(
            X,
            Y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
        )

        # make a saved copy for later analysis
        if not os.path.exists("tfsaves"):
            os.mkdir("tfsaves")
        self.save("tfsaves/fit_model.tf")
        # pylint: disable=attribute-defined-outside-init
        self.saved_was_dpused, self.saved_reason = check_DP_used(self.optimizer)
        self.saved_epsilon = self.current_epsilon
        return returnval

    def posthoc_check(self, verbose: bool = True) -> Tuple[str, bool]:
        """Checks whether model should be considered unsafe
        for example, has been changed since fit() was last run,
        or does not meet DP policy
        """

        disclosive = False
        msg = ""

        # have the model architecture or weights been changed?
        self.save("tfsaves/requested_model.tf")
        models_same, same_msg = check_checkpoint_equality(
            "tfsaves/fit_model.tf",
            "tfsaves/requested_model.tf",
        )
        if not models_same:
            msg += same_msg
            disclosive = True

        # was a dp-enbled optimiser provided?
        allowed, allowedmessage = check_optimizer_allowed(self.optimizer)
        if not allowed:
            msg += allowedmessage
            disclosive = True

        # was the dp-optimiser used during fit()
        dpused, dpusedmessage = check_DP_used(self.optimizer)
        if not dpused:
            msg += dpusedmessage
            disclosive = True

        # have values been changed since saved  immediately after fit()?
        if (
            dpused != self.saved_was_dpused
            or dpusedmessage != self.saved_reason
            or self.saved_epsilon != self.current_epsilon
        ):
            msg += get_reporting_string(name="opt_config_changed")
            disclosive = True

        # if not what was the value of epsilon achieved
        eps_met, cur_eps = self.dp_epsilon_met(
            num_examples=self.num_samples,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )
        if not eps_met:
            dpepsilonmessage = get_reporting_string(
                name="epsilon_above_normal",
                current_epsilon=cur_eps,
            )
            if verbose:
                print(
                    get_reporting_string(
                        name="recommend_further_discussion", msg=dpepsilonmessage
                    )
                )
            msg += dpepsilonmessage
            disclosive = True

        if disclosive:
            msg = get_reporting_string(name="recommend_not_release") + msg
            return msg, True

        # passed all the tests!!
        if verbose:
            msg = get_reporting_string(name="recommend_allow_release")
            msg += get_reporting_string(
                name="allow_release_eps_below_max", current_epsilon=cur_eps
            )
        return msg, False

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
            print(get_reporting_string(name="input_filename_with_extension"))
            return

        thename = self.model_save_file.split(".")
        # print(f'in save(), parsed filename is {thename}')
        if len(thename) == 1:
            print(get_reporting_string(name="filename_must_indicate_type"))
            # "file name must indicate type as a suffix")
        else:
            suffix = self.model_save_file.split(".")[-1]

            if suffix in ("h5", "tf"):
                try:
                    tf.keras.models.save_model(
                        self,
                        self.model_save_file,
                        include_optimizer=False,
                        # save_traces=False,
                        save_format=suffix,
                    )
                # pragma:no cover
                except Exception as er:  # pylint:disable=broad-except # pragma:no cover
                    print(  # pragma:no cover
                        get_reporting_string(
                            name="error_saving_file", suffix=suffix, er=er
                        )
                    )
                    # f"saving as a {suffix} file gave this error message:  {er}")
            else:
                print(
                    get_reporting_string(
                        name="suffix_not_supported_for_type", model_type=self.model_type
                    )
                )
                # f"{suffix} file suffix  not supported "
                # f"for models of type {self.model_type}.\n"
