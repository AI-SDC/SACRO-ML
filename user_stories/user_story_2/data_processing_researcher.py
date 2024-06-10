"""SUPPORTING FILE FOR USER STORY 2.

This file is an example of a function created by a researcher that will pre-process a dataset

To use: write a function that will process your input data, and output the processed version

NOTE: in order to work, this function needs to:

    - take a single parameter (the data to be processed)
    - return a dictionary
    - which contains the keys ]
        ['n_features_raw_data', 'x_transformed', 'y_transformed', 'train_indices']
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def process_dataset(data):
    """Create a function that does the data pre-processing for user story 2."""
    # Replace the contents of this function with your pre-processing code

    labels = np.asarray(data["class"])
    data = np.asarray(data.drop(columns=["class"], inplace=False))

    n_features_raw_data = np.shape(data)[1]

    label_enc = LabelEncoder()
    feature_enc = OneHotEncoder()
    x_transformed = feature_enc.fit_transform(data).toarray()
    y_transformed = label_enc.fit_transform(labels)

    row_indices = np.arange(np.shape(x_transformed)[0])

    # This step is not necessary, however it's the simplest way of getting training indices from
    # the data
    # Any method of generating indices of samples to be used for training will work here
    (
        x_train,
        x_test,
        y_train,
        y_test,
        train_indices,
        test_indices,
    ) = train_test_split(  # pylint: disable=unused-variable
        x_transformed,
        y_transformed,
        row_indices,
        test_size=0.5,
        stratify=y_transformed,
        shuffle=True,
    )

    returned = {}
    returned["n_features_raw_data"] = n_features_raw_data
    returned["x_transformed"] = x_transformed
    returned["y_transformed"] = y_transformed
    returned["train_indices"] = train_indices

    return returned
