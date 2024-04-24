"""This module contains unit tests for SafeKerasModel."""

from __future__ import annotations

import os
import shutil
import warnings

import numpy as np
import pytest
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input  # pylint: disable = import-error

from aisdc.safemodel.classifiers import SafeKerasModel, safekeras
from aisdc.safemodel.reporting import get_reporting_string

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


EPOCHS = 1
n_classes = 4
# expected accuracy
# ACC = 0.325 if platform.system() == "Darwin" else 0.3583333492279053
ACC = 0.3583333492279053
# UNSAFE_ACC = 0.325 if platform.system() == "Darwin" else 0.3583333492279053
UNSAFE_ACC = 0.3583333492279053
RES_DIR = "RES"


def clean():
    """Removes unwanted results."""
    shutil.rmtree(RES_DIR)


def cleanup_file(name: str):
    """Removes unwanted files or directory."""
    if os.path.exists(name) and os.path.isfile(name):  # h5
        os.remove(name)
    elif os.path.exists(name) and os.path.isdir(name):  # tf
        shutil.rmtree(name)


def get_data():
    """Returns data for testing."""
    iris = datasets.load_iris()
    xall = np.asarray(iris["data"], dtype=np.float64)
    yall = np.asarray(iris["target"], dtype=np.float64)
    xall = np.vstack([xall, (7, 2.0, 4.5, 1)])
    yall = np.append(yall, n_classes)
    X, Xval, y, yval = train_test_split(
        xall, yall, test_size=0.2, shuffle=True, random_state=12345
    )
    y = tf.one_hot(y, n_classes)
    yval = tf.one_hot(yval, n_classes)
    return X, y, Xval, yval


def make_small_model(num_hidden_layers=2):
    """Make the keras model."""
    # get data
    X, y, Xval, yval = get_data()
    # set seed and kernel initialisers for repeatability
    tf.random.set_seed(12345)
    initializer = tf.keras.initializers.Zeros()
    input_data = Input(shape=X[0].shape)
    xx = Dense(32, activation="relu", kernel_initializer=initializer)(input_data)
    layers_added = 1
    while layers_added < num_hidden_layers:
        xx = Dense(32, activation="relu", kernel_initializer=initializer)(xx)
        layers_added += 1
    output = Dense(n_classes, activation="softmax", kernel_initializer=initializer)(xx)
    model = SafeKerasModel(
        inputs=input_data,
        outputs=output,
        name="test",
        num_samples=X.shape[0],
        epochs=EPOCHS,
    )

    return model, X, y, Xval, yval


def check_init_completed(model: SafeKerasModel):
    """Basic checks for things that happen
    at end of correct init.
    """
    rightname = "KerasModel"
    assert (
        model.model_type == rightname
    ), "failed check for model type being set in init()"
    errstr = (
        "default value of noise_multiplier=0.5"
        "Should have been reset to value 0.7 in rules.json"
    )
    assert model.noise_multiplier == 0.7, errstr


def test_init_variants():
    """Test alternative ways of calling init
    do just with single layer for speed of testing.
    """
    # get data
    X, _, _, _ = get_data()
    # set seed and kernel initialisers for repeatability
    tf.random.set_seed(12345)
    initializer = tf.keras.initializers.Zeros()

    # define model architecture
    input_data = Input(shape=X[0].shape)
    xx = Dense(32, activation="relu", kernel_initializer=initializer)(input_data)
    output = Dense(n_classes, activation="softmax", kernel_initializer=initializer)(xx)

    # standard way
    model = SafeKerasModel(
        inputs=input_data,
        outputs=output,
        name="test",
        num_samples=X.shape[0],
        epochs=EPOCHS,
    )
    check_init_completed(model)

    # inputs and outputs not in kwargs
    model2 = SafeKerasModel(
        input_data,
        output,
        "test",
        num_samples=X.shape[0],
        epochs=EPOCHS,
    )
    check_init_completed(model2)

    # batch size zero
    model3 = SafeKerasModel(
        inputs=input_data,
        outputs=output,
        name="test",
        num_samples=X.shape[0],
        epochs=EPOCHS,
        batch_size=0,
    )
    errstr = "failed to correct batch_size=0 in init"
    assert model3.batch_size == 32, errstr


def test_same_configs():  # pylint: disable=too-many-locals
    """Check whether tests for equal configuration work."""

    model1, X, _, _, _ = make_small_model(num_hidden_layers=1)
    model2, _, _, _, _ = make_small_model(num_hidden_layers=2)
    model2a, _, _, _, _ = make_small_model(num_hidden_layers=2)

    # different numbers of layers
    same1, msg1 = safekeras.same_configs(model1, model2)
    errstr = (
        f"model1 has {len(model1.layers)} layers, but\n"
        f"model2 has {len(model2.layers)} layers\n"
    )
    assert same1 is False, errstr
    correct_msg1 = get_reporting_string(name="different_layer_count")
    errstr = f"msg    was: {msg1}\n" f"should be : {correct_msg1}"
    assert msg1 == correct_msg1, errstr

    # same architecture
    same2, msg2 = safekeras.same_configs(model2, model2a)
    correct_msg2 = get_reporting_string(name="same_ann_config")
    assert msg2 == correct_msg2, (
        rf"should report {correct_msg2}\," f" but   got    {msg2}.\n"
    )
    assert same2 is True, "models are same!"

    # same layers, different widths
    input_data = Input(shape=X[0].shape)
    initializer = tf.keras.initializers.Zeros()
    xx = Dense(64, activation="relu", kernel_initializer=initializer)(input_data)
    output = Dense(n_classes, activation="softmax", kernel_initializer=initializer)(xx)
    model1a = SafeKerasModel(
        inputs=input_data,
        outputs=output,
        name="test",
        num_samples=X.shape[0],
        epochs=EPOCHS,
    )
    check_init_completed(model1a)
    same3, msg3 = safekeras.same_configs(model1, model1a)
    errmsg = "Should report layers have different num nodes"
    assert same3 is False, errmsg
    correct_msg3 = get_reporting_string(name="layer_configs_differ", layer=1, length=1)
    correct_msg3 += get_reporting_string(
        name="param_changed_from_to", key="units", val="32", cur_val=64
    )
    errstr = f"got message: {msg3}\n" rf"expected.  : {correct_msg3}\."
    assert msg3 == correct_msg3, errstr


def test_same_weights():  # pylint : disable=too-many-locals
    """Check the same weights method catches differences."""
    # make models to test
    model1, X, _, _, _ = make_small_model(num_hidden_layers=1)
    model2, _, _, _, _ = make_small_model(num_hidden_layers=2)
    input_data = Input(shape=X[0].shape)
    initializer = tf.keras.initializers.Zeros()
    xx = Dense(64, activation="relu", kernel_initializer=initializer)(input_data)
    output = Dense(n_classes, activation="softmax", kernel_initializer=initializer)(xx)
    model1a = SafeKerasModel(
        inputs=input_data,
        outputs=output,
        name="test",
        num_samples=X.shape[0],
        epochs=EPOCHS,
    )

    # same
    same1, _ = safekeras.same_weights(model1, model1)
    assert same1 is True

    # different num layers
    same2, _ = safekeras.same_weights(model1, model2)
    assert same2 is False

    # different sized layers
    same3, _ = safekeras.same_weights(model1, model1a)
    errstr = (
        "model1 hidden layer has "
        f"  {len(model1.layers[1].get_weights()[0][0])} units"
        f" but model2 has {len(model1a.layers[1].get_weights()[0][0])}.\n"
    )
    assert same3 is False, errstr


def test_DP_optimizer_checks():
    """Tests the various checks that DP optimiser was used."""
    # make model
    model1, _, _, _, _ = make_small_model(num_hidden_layers=1)
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )

    # compile with allowed optimizer and test check works for positive case
    allowed_optimizers = [
        "tensorflow_privacy.DPKerasAdagradOptimizer",
        "tensorflow_privacy.DPKerasAdamOptimizer",
        "tensorflow_privacy.DPKerasSGDOptimizer",
    ]
    for oktype in allowed_optimizers:
        model, _, _, _, _ = make_small_model(num_hidden_layers=1)
        model.compile(loss=loss, optimizer=oktype)
        opt_ok, msg = safekeras.check_optimizer_allowed(model.optimizer)
        assert opt_ok is True, msg
        opt_is_dp, _ = safekeras.check_optimizer_is_DP(model.optimizer)
        assert opt_is_dp is True

    # not ok optimizer
    model, _, _, _, _ = make_small_model(num_hidden_layers=1)
    model.compile(loss=loss)
    # reset to not DP optimizer
    model.optimizer = tf.keras.optimizers.get("SGD")
    opt_ok, msg = safekeras.check_optimizer_allowed(model1.optimizer)
    assert opt_ok is False, msg
    opt_is_dp, msg = safekeras.check_optimizer_is_DP(model1.optimizer)
    assert opt_is_dp is False, msg


def test_DP_used():
    """Tests the various checks that DP optimiser was used."""
    # should pass aftyer model compiled **and** fitted with DP optimizer
    model1, X, y, Xval, yval = make_small_model(num_hidden_layers=1)
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model1.compile(loss=loss)
    dp_used, msg = safekeras.check_DP_used(model1.optimizer)
    assert dp_used is False
    model1.fit(X, y, validation_data=(Xval, yval), epochs=EPOCHS, batch_size=20)
    dp_used, msg = safekeras.check_DP_used(model1.optimizer)
    assert dp_used is True

    # this model gets changed to non-DP by calling the superclass compile()
    # so should fail all the checks
    model2, _, _, _, _ = make_small_model(num_hidden_layers=1)
    super(SafeKerasModel, model2).compile(loss=loss, optimizer="SGD")
    model2.fit(X, y, validation_data=(Xval, yval), epochs=EPOCHS, batch_size=20)
    dp_used, msg = safekeras.check_DP_used(model2.optimizer)
    assert dp_used is False, msg


def test_checkpoints_are_equal():
    """Test the check for checkpoint equality."""
    model1, X, y, Xval, yval = make_small_model(num_hidden_layers=1)
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model1.compile(loss=loss)
    model1.fit(X, y, validation_data=(Xval, yval), epochs=EPOCHS, batch_size=20)
    model1.save("fit.tf")
    model1.fit(X, y, validation_data=(Xval, yval), epochs=EPOCHS * 2, batch_size=20)
    model1.save("refit.tf")

    # same arch, different weights
    same, msg = safekeras.check_checkpoint_equality("fit.tf", "refit.tf")
    assert same is False, msg

    # should be same
    same, msg = safekeras.check_checkpoint_equality("fit.tf", "fit.tf")
    print(msg)
    assert same is True, msg

    # different architecture
    model2, X, y, Xval, yval = make_small_model(num_hidden_layers=3)
    model2.compile(loss=loss)
    model2.fit(X, y, validation_data=(Xval, yval), epochs=EPOCHS, batch_size=20)
    model2.save("fit2.tf")

    same, msg = safekeras.check_checkpoint_equality("fit.tf", "fit2.tf")
    print(msg)
    assert same is False, msg

    # coping with trashed files
    cleanup_file("fit.tf/saved_model.pb")
    same, msg = safekeras.check_checkpoint_equality("fit.tf", "fit2.tf")
    assert same is False, msg
    same, msg = safekeras.check_checkpoint_equality("fit2.tf", "fit.tf")
    assert same is False, msg

    same, msg = safekeras.check_checkpoint_equality("hello", "fit2.tf")
    assert same is False
    assert "Error re-loading  model from" in msg

    same, msg = safekeras.check_checkpoint_equality("fit2.tf", "hello")
    assert same is False
    assert "Error re-loading  model from" in msg

    for name in ("fit.tf", "fit2.tf", "refit.tf"):
        cleanup_file(name)


def test_load():
    """Tests the oading functionality."""

    # make a model, train then save it
    model, X, y, Xval, yval = make_small_model()
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)
    model.fit(X, y, validation_data=(Xval, yval), epochs=EPOCHS, batch_size=20)
    model.save("keras_save.tf")

    # won't load with invalid names
    ok, _ = safekeras.load_safe_keras_model()
    assert ok is False, "can't load with no model file name"

    ok, _ = safekeras.load_safe_keras_model("keras_save.h5")
    assert ok is False, "can only load from .tf file"

    # should load fine with right name
    ok, reloaded_model = safekeras.load_safe_keras_model("keras_save.tf")
    assert ok is True
    ypred = "over-write-me"
    ypred = reloaded_model.predict(X)
    assert isinstance(ypred, np.ndarray)

    cleanup_file("keras_save.tf")
    cleanup_file("tfsaves")


def test_keras_model_created():
    """Test keras model."""
    model, _, _, _, _ = make_small_model()
    rightname = "KerasModel"
    assert (
        model.model_type == rightname
    ), "failed check for model type being set in init()"
    # noise multiplier should have been reset from default to one that matches rules.json
    assert model.noise_multiplier == 0.7


def test_second_keras_model_created():
    """Test second keras model."""
    X, _, _, _ = get_data()
    tf.random.set_seed(12345)
    initializer = tf.keras.initializers.Zeros()
    input_data = Input(shape=X[0].shape)
    xx = Dense(128, activation="relu", kernel_initializer=initializer)(input_data)
    xx = Dense(128, activation="relu", kernel_initializer=initializer)(xx)
    xx = Dense(64, activation="relu", kernel_initializer=initializer)(xx)
    output = Dense(n_classes, activation="softmax", kernel_initializer=initializer)(xx)
    _ = SafeKerasModel(
        inputs=input_data,
        outputs=output,
        name="test",
        num_samples=X.shape[0],
        epochs=EPOCHS,
    )
    model2 = SafeKerasModel(
        inputs=input_data,
        outputs=output,
        name="test",
        num_samples=X.shape[0],
        epochs=EPOCHS,
    )
    rightname = "KerasModel"
    assert (
        model2.model_type == rightname
    ), "failed check for second model type being set in init()"
    # noise multiplier should have been reset from default to one that matches rules.json
    assert model2.noise_multiplier == 0.7


def test_keras_model_compiled_as_DP():
    """Test Compile DP."""
    model, X, _, _, _ = make_small_model()
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)
    isDP, _ = safekeras.check_optimizer_is_DP(model.optimizer)
    assert isDP, "failed check that optimizer is dP"

    right_epsilon = 20.363059561511612
    model.check_epsilon(X.shape[0], 20, 10)
    assert model.current_epsilon == pytest.approx(
        right_epsilon, 0.01
    ), "failed check for epsilon after compilation"

    # check this works
    ok, msg = model.check_epsilon(X.shape[0], 0, 10)
    assert ok, "should be ok after resetting batch size =1"
    correct_msg = get_reporting_string(name="division_by_zero")
    assert msg == correct_msg, msg


def test_keras_basic_fit():
    """SafeKeras using recommended values."""
    model, X, y, Xval, yval = make_small_model()

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)

    isDP, msg = safekeras.check_optimizer_is_DP(model.optimizer)
    assert isDP, "failed check that optimizer is dP"

    # first check that stops when not DP  if they say refine
    ok, msg = model.fit(
        X,
        y,
        validation_data=(Xval, yval),
        epochs=10,
        batch_size=X.shape[0],
        refine_epsilon=True,
    )
    assert ok is False

    # now default (False)
    model.fit(X, y, validation_data=(Xval, yval), epochs=EPOCHS, batch_size=20)

    DPused, msg = safekeras.check_DP_used(model.optimizer)
    assert (
        DPused
    ), "Failed check that DP version of optimiser was actually used in training"

    loss, acc = model.evaluate(X, y)
    expected_accuracy = UNSAFE_ACC
    assert round(acc, 6) == round(
        expected_accuracy, 6
    ), "failed check that accuracy is as expected"

    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg, "failed check params are within range"
    assert disclosive is False, "failed check disclosive is false"


def test_keras_save_actions():
    """Test save action."""
    # create, compile and train model
    model, X, y, Xval, yval = make_small_model()

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)
    model.fit(X, y, validation_data=(Xval, yval), epochs=EPOCHS, batch_size=20)

    # start with .tf and .h5 which should work
    names = ("safekeras.tf", "safekeras.h5")
    for name in names:
        # clear existing files
        cleanup_file(name)
        # save file
        model.save(name)
        assert os.path.exists(name), f"Failed test to save model as {name}"
        # clean up
        cleanup_file(name)

    # now other versions which should not
    names = ("safekeras.sav", "safekeras.pkl", "randomfilename", "undefined")
    for name in names:
        cleanup_file(name)
        model.save(name)
        assert os.path.exists(name) is False, f"Failed test NOT to save model as {name}"
        cleanup_file(name)
    # cleeanup
    cleanup_file("tfsaves")


def test_keras_unsafe_l2_norm():
    """SafeKeras using unsafe values."""
    model, X, y, Xval, yval = make_small_model()

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)

    isDP, msg = safekeras.check_optimizer_is_DP(model.optimizer)
    assert isDP, "failed check that optimizer is dP"

    model.l2_norm_clip = 0.9

    model.fit(X, y, validation_data=(Xval, yval), epochs=EPOCHS, batch_size=20)

    DPused, msg = safekeras.check_DP_used(model.optimizer)
    assert (
        DPused
    ), "Failed check that DP version of optimiser was actually used in training"

    loss, acc = model.evaluate(X, y)
    expected_accuracy = UNSAFE_ACC
    assert round(acc, 6) == round(
        expected_accuracy, 6
    ), "failed check that accuracy is as expected"

    msg, disclosive = model.preliminary_check()
    correct_msg = (
        "WARNING: model parameters may present a disclosure risk:"
        "\n- parameter l2_norm_clip = 0.9 identified as less than the recommended "
        "min value of 1.0."
    )
    assert msg == correct_msg, "failed check correct warning message"
    assert disclosive is True, "failed check disclosive is True"


def test_keras_unsafe_noise_multiplier():
    """SafeKeras using unsafe values."""
    model, X, y, Xval, yval = make_small_model()

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)

    isDP, msg = safekeras.check_optimizer_is_DP(model.optimizer)
    assert isDP, "failed check that optimizer is dP"

    model.noise_multiplier = 1.0

    model.fit(X, y, validation_data=(Xval, yval), epochs=EPOCHS, batch_size=20)

    DPused, msg = safekeras.check_DP_used(model.optimizer)
    assert (
        DPused
    ), "Failed check that DP version of optimiser was actually used in training"

    loss, acc = model.evaluate(X, y)
    expected_accuracy = UNSAFE_ACC
    assert round(acc, 6) == round(
        expected_accuracy, 6
    ), "failed check that accuracy is as expected"

    msg, disclosive = model.preliminary_check()
    correct_msg = (
        "WARNING: model parameters may present a disclosure risk:"
        "\n- parameter noise_multiplier = 1.0 identified as greater than the "
        "recommended max value of 0.9."
    )

    assert msg == correct_msg, "failed check params are within range"
    assert disclosive is True, "failed check disclosive is True"


def test_keras_unsafe_min_epsilon():
    """SafeKeras using unsafe values."""
    model, X, y, Xval, yval = make_small_model()

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)

    isDP, msg = safekeras.check_optimizer_is_DP(model.optimizer)
    assert isDP, "failed check that optimizer is dP"

    model.min_epsilon = 4

    model.fit(X, y, validation_data=(Xval, yval), epochs=EPOCHS, batch_size=20)

    DPused, msg = safekeras.check_DP_used(model.optimizer)
    assert (
        DPused
    ), "Failed check that DP version of optimiser was actually used in training"

    loss, acc = model.evaluate(X, y)
    expected_accuracy = UNSAFE_ACC
    assert round(acc, 6) == round(
        expected_accuracy, 6
    ), "failed check that accuracy is as expected"

    msg, disclosive = model.preliminary_check()
    correct_msg = (
        "WARNING: model parameters may present a disclosure risk:"
        "\n- parameter min_epsilon = 4 identified as less than the recommended min value of 5."
    )

    assert msg == correct_msg, "failed check correct warning message"
    assert disclosive is True, "failed check disclosive is True"


def test_keras_unsafe_delta():
    """SafeKeras using unsafe values."""
    model, X, y, Xval, yval = make_small_model()

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)

    isDP, msg = safekeras.check_optimizer_is_DP(model.optimizer)
    assert isDP, "failed check that optimizer is dP"

    model.delta = 1e-6

    model.fit(X, y, validation_data=(Xval, yval), epochs=EPOCHS, batch_size=20)

    DPused, msg = safekeras.check_DP_used(model.optimizer)
    assert (
        DPused
    ), "Failed check that DP version of optimiser was actually used in training"

    loss, acc = model.evaluate(X, y)
    expected_accuracy = UNSAFE_ACC
    assert round(acc, 6) == round(
        expected_accuracy, 6
    ), "failed check that accuracy is as expected"

    msg, disclosive = model.preliminary_check()
    correct_msg = (
        "WARNING: model parameters may present a disclosure risk:\n"
        "- parameter delta = 1e-06 identified as less than the recommended min value of 1e-05."
    )
    assert msg == correct_msg, "failed check params are within range"
    assert disclosive is True, "failed check disclosive is True"


def test_keras_unsafe_batch_size():
    """SafeKeras using unsafe values."""
    model, X, y, Xval, yval = make_small_model()

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)

    isDP, msg = safekeras.check_optimizer_is_DP(model.optimizer)
    assert isDP, "failed check that optimizer is dP"

    model.batch_size = 34

    model.fit(X, y, validation_data=(Xval, yval), epochs=EPOCHS, batch_size=20)

    DPused, msg = safekeras.check_DP_used(model.optimizer)
    assert (
        DPused
    ), "Failed check that DP version of optimiser was actually used in training"

    loss, acc = model.evaluate(X, y)
    expected_accuracy = UNSAFE_ACC
    assert round(acc, 6) == round(
        expected_accuracy, 6
    ), "failed check that accuracy is as expected"

    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg, "failed check params are within range"
    assert disclosive is False, "failed check disclosive is false"


def test_keras_unsafe_learning_rate():
    """SafeKeras using unsafe values."""
    model, X, y, Xval, yval = make_small_model()

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)

    isDP, msg = safekeras.check_optimizer_is_DP(model.optimizer)
    assert isDP, "failed check that optimizer is dP"

    model.learning_rate = 0.2

    model.fit(X, y, validation_data=(Xval, yval), epochs=EPOCHS, batch_size=20)

    DPused, msg = safekeras.check_DP_used(model.optimizer)
    assert (
        DPused
    ), "Failed check that DP version of optimiser was actually used in training"

    loss, acc = model.evaluate(X, y)
    expected_accuracy = UNSAFE_ACC
    assert round(acc, 6) == round(
        expected_accuracy, 6
    ), "failed check that accuracy is as expected"

    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"

    assert msg == correct_msg, "failed check warning message incorrect"
    assert disclosive is False, "failed check disclosive is false"


def test_create_checkfile():
    """Test create checkfile."""
    # create, compile and train model
    model, X, y, Xval, yval = make_small_model()

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)
    model.fit(X, y, validation_data=(Xval, yval), epochs=EPOCHS, batch_size=20)

    # start with .tf and .h5 which should work
    exts = ("tf", "h5")
    for ext in exts:
        name = os.path.normpath(f"{RES_DIR}/model.{ext}")
        os.makedirs(os.path.dirname(name), exist_ok=True)
        # check save file
        model.save(name)
        assert os.path.exists(name), f"Failed test to save model as {name}"
        clean()
        # check release
        model.request_release(path=RES_DIR, ext=ext)
        assert os.path.exists(name), f"Failed test to save model as {name}"
        name = os.path.normpath(f"{RES_DIR}/target.json")
        assert os.path.exists(name), "Failed test to save target.json"
        clean()

    # now other versions which should not
    exts = ("sav", "pkl", "undefined")
    for ext in exts:
        name = os.path.normpath(f"{RES_DIR}/model.{ext}")
        os.makedirs(os.path.dirname(name), exist_ok=True)
        model.save(name)
        assert os.path.exists(name) is False, f"Failed test NOT to save model as {name}"
        clean()


def test_posthoc_check():
    """Testing the posthoc checking function."""
    # make a model, train then save it
    model, X, y, Xval, yval = make_small_model()
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)
    model.fit(X, y, validation_data=(Xval, yval), epochs=1, batch_size=20)

    # should be ok
    _, disclosive = model.posthoc_check()
    assert disclosive is False, "base config in tests should be ok"

    # change optimizer and some other settings
    # in way that stresses lots of routes
    cleanup_file("tfsaves/fit_model.tf")
    model.epochs = 1000
    model.optimizer = tf.keras.optimizers.get("SGD")
    _, disclosive = model.posthoc_check()
    assert disclosive is True, "should pick up optimizer changed"

    cleanup_file("keras_save.tf")
    cleanup_file("tfsaves")


def test_final_cleanup():
    """Clean up any files let around by other tests."""
    cleanup_file("tfsaves")
