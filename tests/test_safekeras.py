"""This module contains unit tests for SafeKerasModel."""

import os
import platform
import shutil
import getpass

import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input # pylint: disable = import-error, no-name-in-module

from safemodel.classifiers import SafeKerasModel

n_classes = 4
#expected accuracy
ACC = 0.6750 if platform.system()== "Darwin" else  0.3583333492279053

def cleanup_file(name:str):
    """removes unwanted files or directory"""
    if os.path.exists(name) and os.path.isfile(name):  # h5
        os.remove(name)
    elif os.path.exists(name) and os.path.isdir(name):  # tf
        shutil.rmtree(name)

def get_data():
    """Returns data for testing."""
    iris = datasets.load_iris()
    xall = np.asarray(iris['data'], dtype=np.float64)
    yall = np.asarray(iris['target'], dtype=np.float64)
    xall = np.vstack([xall, (7, 2.0, 4.5, 1)])
    yall = np.append(yall, n_classes)
    X, Xval, y, yval = train_test_split(
        xall, yall, test_size=0.2, shuffle=True, random_state=12345
    )
    y = tf.one_hot(y, n_classes)
    yval = tf.one_hot(yval, n_classes)
    return X, y, Xval, yval


def make_model():
    "Make the keras model"
    # get data
    X, y, Xval, yval = get_data()
    # set seed and kernel initialisers for repeatability
    tf.random.set_seed(12345)
    initializer = tf.keras.initializers.Zeros()

    input_data = Input(shape=X[0].shape)
    xx = Dense(128, activation="relu", kernel_initializer=initializer)(input_data)
    xx = Dense(128, activation="relu", kernel_initializer=initializer)(xx)
    xx = Dense(64, activation="relu", kernel_initializer=initializer)(xx)
    output = Dense(n_classes, activation="softmax", kernel_initializer=initializer)(xx)
    model = SafeKerasModel(
        inputs=input_data,
        outputs=output,
        name="test",
        num_samples=X.shape[0],
        epochs=10,
    )

    return model, X, y, Xval, yval


def test_keras_model_created():
    "Test keras model"
    model, _, _, _, _ = make_model()
    rightname = "KerasModel"
    assert (
        model.model_type == rightname
    ), "failed check for model type being set in init()"
    # noise multiplier should have been reset from default to one that matches rules.json
    assert model.noise_multiplier == 0.7


def test_second_keras_model_created():
    "Test second keras model"
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
        epochs=10,
    )
    model2 = SafeKerasModel(
        inputs=input_data,
        outputs=output,
        name="test",
        num_samples=X.shape[0],
        epochs=10,
    )
    rightname = "KerasModel"
    assert (
        model2.model_type == rightname
    ), "failed check for second model type being set in init()"
    # noise multiplier should have been reset from default to one that matches rules.json
    assert model2.noise_multiplier == 0.7


def test_keras_model_compiled_as_DP():
    "Test Compile DP"
    model, X, _, _, _ = make_model()
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)
    isDP, _ = model.check_optimizer_is_DP(model.optimizer)
    assert isDP, "failed check that optimizer is dP"

    right_epsilon = 20.363059561511612
    model.check_epsilon(X.shape[0], 20, 10)
    assert (
        model.current_epsilon == right_epsilon
    ), "failed check for epsilon after compilation"


def test_keras_basic_fit():
    """SafeKeras using recommended values."""
    model, X, y, Xval, yval = make_model()

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)

    isDP, msg = model.check_optimizer_is_DP(model.optimizer)
    assert isDP, "failed check that optimizer is dP"

    model.fit(X, y, validation_data=(Xval, yval), epochs=10, batch_size=20)

    DPused, msg = model.check_DP_used(model.optimizer)
    assert (
        DPused
    ), "Failed check that DP version of optimiser was actually used in training"

    loss, acc = model.evaluate(X, y)
    expected_accuracy = ACC
    assert round(acc,6) == round(expected_accuracy,6), "failed check that accuracy is as expected"

    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg, "failed check params are within range"
    assert disclosive is False, "failed check disclosive is false"


def test_keras_save_actions():
    "Test save action"
    # create, compile and train model
    model, X, y, Xval, yval = make_model()

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)
    model.fit(X, y, validation_data=(Xval, yval), epochs=10, batch_size=20)

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
    names = ("safekeras.sav", "safekeras.pkl", "randomfilename")
    for name in names:
        cleanup_file(name)
        model.save(name)
        assert os.path.exists(name) is False, f"Failed test NOT to save model as {name}"
        cleanup_file(name)


def test_keras_unsafe_l2_norm():
    """SafeKeras using unsafe values."""
    model, X, y, Xval, yval = make_model()

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)

    isDP, msg = model.check_optimizer_is_DP(model.optimizer)
    assert isDP, "failed check that optimizer is dP"

    model.l2_norm_clip = 0.9

    model.fit(X, y, validation_data=(Xval, yval), epochs=10, batch_size=20)

    DPused, msg = model.check_DP_used(model.optimizer)
    assert (
        DPused
    ), "Failed check that DP version of optimiser was actually used in training"

    loss, acc = model.evaluate(X, y)
    expected_accuracy = ACC
    assert round(acc,6) == round(expected_accuracy,6), "failed check that accuracy is as expected"

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
    model, X, y, Xval, yval = make_model()

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)

    isDP, msg = model.check_optimizer_is_DP(model.optimizer)
    assert isDP, "failed check that optimizer is dP"

    model.noise_multiplier = 1.0

    model.fit(X, y, validation_data=(Xval, yval), epochs=10, batch_size=20)

    DPused, msg = model.check_DP_used(model.optimizer)
    assert (
        DPused
    ), "Failed check that DP version of optimiser was actually used in training"

    loss, acc = model.evaluate(X, y)
    expected_accuracy = ACC
    assert round(acc,6) == round(expected_accuracy,6), "failed check that accuracy is as expected"

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
    model, X, y, Xval, yval = make_model()

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)

    isDP, msg = model.check_optimizer_is_DP(model.optimizer)
    assert isDP, "failed check that optimizer is dP"

    model.min_epsilon = 4

    model.fit(X, y, validation_data=(Xval, yval), epochs=10, batch_size=20)

    DPused, msg = model.check_DP_used(model.optimizer)
    assert (
        DPused
    ), "Failed check that DP version of optimiser was actually used in training"

    loss, acc = model.evaluate(X, y)
    expected_accuracy = ACC
    assert round(acc,6) == round(expected_accuracy,6), "failed check that accuracy is as expected"

    msg, disclosive = model.preliminary_check()
    correct_msg = (
        "WARNING: model parameters may present a disclosure risk:"
        "\n- parameter min_epsilon = 4 identified as less than the recommended min value of 5."
    )

    assert msg == correct_msg, "failed check correct warning message"
    assert disclosive is True, "failed check disclosive is True"

def test_keras_unsafe_delta():
    """SafeKeras using unsafe values."""
    model, X, y, Xval, yval = make_model()

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)

    isDP, msg = model.check_optimizer_is_DP(model.optimizer)
    assert isDP, "failed check that optimizer is dP"

    model.delta = 1e-6

    model.fit(X, y, validation_data=(Xval, yval), epochs=10, batch_size=20)

    DPused, msg = model.check_DP_used(model.optimizer)
    assert (
        DPused
    ), "Failed check that DP version of optimiser was actually used in training"

    loss, acc = model.evaluate(X, y)
    expected_accuracy = ACC
    assert round(acc,6) == round(expected_accuracy,6), "failed check that accuracy is as expected"

    msg, disclosive = model.preliminary_check()
    correct_msg = (
        "WARNING: model parameters may present a disclosure risk:\n"
        "- parameter delta = 1e-06 identified as less than the recommended min value of 1e-05."
    )
    assert msg == correct_msg, "failed check params are within range"
    assert disclosive is True, "failed check disclosive is True"

def test_keras_unsafe_batch_size():
    """SafeKeras using unsafe values."""
    model, X, y, Xval, yval = make_model()

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)

    isDP, msg = model.check_optimizer_is_DP(model.optimizer)
    assert isDP, "failed check that optimizer is dP"

    model.batch_size = 34

    model.fit(X, y, validation_data=(Xval, yval), epochs=10, batch_size=20)

    DPused, msg = model.check_DP_used(model.optimizer)
    assert (
        DPused
    ), "Failed check that DP version of optimiser was actually used in training"

    loss, acc = model.evaluate(X, y)
    expected_accuracy = ACC
    assert round(acc,6) == round(expected_accuracy,6), "failed check that accuracy is as expected"

    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg, "failed check params are within range"
    assert disclosive is False, "failed check disclosive is false"


def test_keras_unsafe_learning_rate():
    """SafeKeras using unsafe values."""
    model, X, y, Xval, yval = make_model()

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)

    isDP, msg = model.check_optimizer_is_DP(model.optimizer)
    assert isDP, "failed check that optimizer is dP"

    model.learning_rate = 0.2

    model.fit(X, y, validation_data=(Xval, yval), epochs=10, batch_size=20)

    DPused, msg = model.check_DP_used(model.optimizer)
    assert (
        DPused
    ), "Failed check that DP version of optimiser was actually used in training"

    loss, acc = model.evaluate(X, y)
    expected_accuracy = ACC
    assert round(acc,6) == round(expected_accuracy,6), "failed check that accuracy is as expected"

    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"

    assert msg == correct_msg, "failed check warning message incorrect"
    assert disclosive is False, "failed check disclosive is false"

def test_create_checkfile():
    """Test create checkfile"""
    # create, compile and train model
    model, X, y, Xval, yval = make_model()

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)
    model.fit(X, y, validation_data=(Xval, yval), epochs=10, batch_size=20)

    # start with .tf and .h5 which should work
    names = ("safekeras.tf", "safekeras.h5")
    for name in names:
        # clear existing files
        cleanup_file(name)
        # save file
        model.save(name)
        assert os.path.exists(name), f"Failed test to save model as {name}"

        model.request_release(name)
        assert os.path.exists(name), f"Failed test to save model as {name}"

        researcher = getpass.getuser()
        outputfilename = researcher + "_checkfile.json"
        assert os.path.exists(outputfilename), f"Failed test to save checkfile as {outputfilename}"

        # Using readlines()
        with open(outputfilename, 'r', encoding='utf-8') as file1:
            lines = file1.readlines()

        count = 0
        # Strips the newline character
        for line in lines:
            count += 1
            print(f"Line{count}: {line.strip()}")

        # clean up
        cleanup_file(name)

    # now other versions which should not
    names = ("safekeras.sav", "safekeras.pkl", "randomfilename")
    for name in names:
        cleanup_file(name)
        model.save(name)
        assert os.path.exists(name) is False, f"Failed test NOT to save model as {name}"
        cleanup_file(name)
