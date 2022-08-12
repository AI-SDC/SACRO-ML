"""This module contains unit tests for SafeKerasModel."""

import os
import pickle
import shutil

import joblib
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model

from safemodel.classifiers import SafeKerasModel

n_classes = 4


def get_data():
    """Returns data for testing."""
    iris = datasets.load_iris()
    xall = np.asarray(iris.data, dtype=np.float64)
    yall = np.asarray(iris.target, dtype=np.float64)
    xall = np.vstack([xall, (7, 2.0, 4.5, 1)])
    yall = np.append(yall, n_classes)
    X, Xval, y, yval = train_test_split(
        xall, yall, test_size=0.2, shuffle=True, random_state=12345
    )
    y = tf.one_hot(y, n_classes)
    yval = tf.one_hot(yval, n_classes)
    return X, y, Xval, yval


def make_model():
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
    model, X, y, Xval, yval = make_model()
    rightname = "KerasModel"
    assert (
        model.model_type == rightname
    ), "failed check for model type being set in init()"
    # noise multiplier should have been reset from default to one that matches rules.json
    assert model.noise_multiplier == 0.7


def test_second_keras_model_created():
    X, y, Xval, yval = get_data()
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
    model, X, y, Xval, yval = make_model()
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    model.compile(loss=loss, optimizer=None)
    isDP, msg = model.check_optimizer_is_DP(model.optimizer)
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
    expected_accuracy = 0.3583333492279053
    assert round(acc,6) == round(expected_accuracy,6), "failed check that accuracy is as expected"

    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg, "failed check params are within range"
    assert disclosive is False, "failed check disclosive is false"


def test_keras_save_actions():
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
        if os.path.exists(name) and os.path.isfile(name):  # h5
            os.remove(name)
        elif os.path.exists(name) and os.path.isdir(name):  # tf
            shutil.rmtree(name)
        # save file
        model.save(name)
        assert os.path.exists(name), f"Failed test to save model as {name}"
        # clean up
        if os.path.isfile(name):  # h5
            os.remove(name)
        elif os.path.isdir(name):
            shutil.rmtree(name)

    # now other versions which should not
    names = ("safekeras.sav", "safekeras.pkl", "randomfilename")
    for name in names:
        if os.path.exists(name):
            os.remove(name)
        model.save(name)
        assert os.path.exists(name) == False, f"Failed test NOT to save model as {name}"
        if os.path.exists(name):
            os.remove(name)
