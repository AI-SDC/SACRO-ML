import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def process_dataset(data):
    y = np.asarray(data["class"])
    x = np.asarray(data.drop(columns=["class"], inplace=False))

    n_features_raw_data = np.shape(x)[1]

    label_enc = LabelEncoder()
    feature_enc = OneHotEncoder()
    x_transformed = feature_enc.fit_transform(x).toarray()
    y_transformed = label_enc.fit_transform(y)

    row_indices = np.arange(np.shape(x_transformed)[0])

    (
        x_train,
        x_test,
        y_train,
        y_test,
        train_indices,
        test_indices,
    ) = train_test_split(
        x_transformed,
        y_transformed,
        row_indices,
        test_size=0.5,
        stratify=y_transformed,
        shuffle=True,
    )

    returned = {}
    returned['n_features_raw_data'] = n_features_raw_data
    returned["x_transformed"] = x_transformed
    returned["y_transformed"] = y_transformed
    returned["train_indices"] = train_indices

    return returned
