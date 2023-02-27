""" code to test the file attacks/dataset
Jim Smith james.smith@uwe.ac.uk 2022
"""
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from aisdc.attacks.dataset import Data


def test_dataset():  # pylint:disable=too-many-locals
    """returns a randomly sampled 10+10% of
    the nursery data set as a Data object
    if needed fetches it from openml and saves. it

    """

    nursery_data = fetch_openml(data_id=26, as_frame=True)
    x = np.asarray(nursery_data.data, dtype=str)
    y = np.asarray(nursery_data.target, dtype=str)
    # change labels from recommend to priority for the two odd cases
    num = len(y)
    for i in range(num):
        if y[i] == "recommend":
            y[i] = "priority"

    indices: list[list[int]] = [
        [0, 1, 2],  # parents
        [3, 4, 5, 6, 7],  # has_nurs
        [8, 9, 10, 11],  # form
        [12, 13, 14, 15],  # children
        [16, 17, 18],  # housing
        [19, 20],  # finance
        [21, 22, 23],  # social
        [24, 25, 26],  # health
        [27],  # dummy
    ]

    # [Researcher] Split into training and test sets
    # target model train / test split - these are strings
    (
        x_train_orig,
        x_test_orig,
        y_train_orig,
        y_test_orig,
    ) = train_test_split(
        x,
        y,
        test_size=0.05,
        stratify=y,
        shuffle=True,
    )

    # now resample the training data reduce number of examples
    _, x_train_orig, _, y_train_orig = train_test_split(
        x_train_orig,
        y_train_orig,
        test_size=0.05,
        stratify=y_train_orig,
        shuffle=True,
    )

    # [Researcher] Preprocess dataset
    # one-hot encoding of features and integer encoding of labels
    label_enc = LabelEncoder()
    feature_enc = OneHotEncoder()
    x_train = feature_enc.fit_transform(x_train_orig).toarray()
    y_train = label_enc.fit_transform(y_train_orig)
    x_test = feature_enc.transform(x_test_orig).toarray()
    y_test = label_enc.transform(y_test_orig)

    # add dummy continuous valued attribute
    dummy_tr = np.random.rand(x_train.shape[0], 1)
    dummy_te = np.random.rand(x_test.shape[0], 1)
    x_train = np.hstack((x_train, dummy_tr))
    x_train_orig = np.hstack((x_train_orig, dummy_tr))
    x_test = np.hstack((x_test, dummy_te))
    x_test_orig = np.hstack((x_test_orig, dummy_te))

    n_features = np.shape(x_train_orig)[1]

    # [TRE / Researcher] Wrap the data in a dataset object
    the_data = Data()
    the_data.name = "nursery"
    the_data.add_processed_data(x_train, y_train, x_test, y_test)
    the_data.add_raw_data(x, y, x_train_orig, y_train_orig, x_test_orig, y_test_orig)
    for i in range(n_features - 1):
        the_data.add_feature(nursery_data.feature_names[i], indices[i], "onehot")
    the_data.add_feature("dummy", indices[n_features - 1], "float")
