import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def process_dataset(data):
    y = np.asarray(data["class"])
    x = np.asarray(data.drop(columns=["class"], inplace=False))

    label_enc = LabelEncoder()
    feature_enc = OneHotEncoder()
    x_transformed = feature_enc.fit_transform(x).toarray()
    y_transformed = label_enc.fit_transform(y)

    indices: list[list[int]] = [
        [0, 1, 2],  # parents
        [3, 4, 5, 6, 7],  # has_nurs
        [8, 9, 10, 11],  # form
        [12, 13, 14, 15],  # children
        [16, 17, 18],  # housing
        [19, 20],  # finance
        [21, 22, 23],  # social
        [24, 25, 26],  # health
    ]

    # Preprocess dataset
    # one-hot encoding of features and integer encoding of label

    returned = {}
    returned['x_transformed'] = x_transformed
    returned['y_transformed'] = y_transformed
    returned['indices'] = indices

    return returned