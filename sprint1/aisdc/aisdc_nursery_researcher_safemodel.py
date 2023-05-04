"""
This module presents example model training a TRE researcher may perform.
Uses SafeRandomForestClassifier from aisdc.
"""

import os
import pandas as pd
import numpy as np

from scipy.io.arff import loadarff
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from aisdc.attacks.dataset import Data
from aisdc.safemodel.classifiers import SafeRandomForestClassifier

DIR = "training_artefacts/"
print("\nCreating directory for training artefacts")

if not os.path.exists(DIR):
    os.makedirs(DIR)

path = os.path.join("../data", "dataset_26_nursery.arff")
data = loadarff(path)
data = pd.DataFrame(data[0])
data = data.select_dtypes([object])
data = data.stack().str.decode("utf-8").unstack()

x = data.drop(columns=["class"], inplace=False)
y = data["class"].values
feature_names = x.columns
x = np.asarray(x)
y = np.asarray(y)

# target model train / test split - these are strings
(
    x_train_orig,
    x_test_orig,
    y_train_orig,
    y_test_orig,
) = train_test_split(
    x,
    y,
    test_size=0.7,
    stratify=y,
    random_state=42,
)

# one-hot encoding of features and integer encoding of labels
label_enc = LabelEncoder()
feature_enc = OneHotEncoder()
trainX = feature_enc.fit_transform(x_train_orig).toarray()
trainy = label_enc.fit_transform(y_train_orig)
testX = feature_enc.transform(x_test_orig).toarray()
testy = label_enc.transform(y_test_orig)

indices = [
    [0, 1, 2],  # parents
    [3, 4, 5, 6, 7],  # has_nurs
    [8, 9, 10, 11],  # form
    [12, 13, 14, 15],  # children
    [16, 17, 18],  # housing
    [19, 20],  # finance
    [21, 22, 23],  # social
    [24, 25, 26],  # health
]

# Wrap the data in a dataset object
sdc_data = Data()
sdc_data.name = "nursery"
sdc_data.add_processed_data(trainX, trainy, testX, testy)
sdc_data.add_raw_data(x, y, x_train_orig, y_train_orig, x_test_orig, y_test_orig)
for i, feature in indices:
    sdc_data.add_feature(feature_names[i], feature, "onehot")

# These hyperparameters lead to a dangerously disclosive trained model
DISCLOSIVE_PARAMS = {}
DISCLOSIVE_PARAMS["min_samples_split"] = 2
DISCLOSIVE_PARAMS["min_samples_leaf"] = 1
DISCLOSIVE_PARAMS["max_depth"] = None
DISCLOSIVE_PARAMS["bootstrap"] = False

print("\nTraining SafeRandomForestClassifier with disclosive hyperparameters:")
print(str(DISCLOSIVE_PARAMS))

# Create and train a SafeRandomForestClassifier
target_model = SafeRandomForestClassifier(**DISCLOSIVE_PARAMS)
target_model.fit(trainX, trainy)
target_model.preliminary_check()

train_acc = accuracy_score(trainy, target_model.predict(trainX))
test_acc = accuracy_score(testy, target_model.predict(testX))

print(f"Training accuracy on SafeRandomForestClassifier: {train_acc:.2f}")
print(f"Testing accuracy on SafeRandomForestClassifier: {test_acc:.2f}")

FILENAME = f"{DIR}/SafeRandomForest.sav"
print(f"\nRequesting release: {FILENAME} and running attacks...\n")
target_model.request_release(FILENAME, sdc_data)
