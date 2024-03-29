"""
This module presents example model training a TRE researcher may perform.
Uses RandomForestClassifier from sklearn.
"""

import os
import pickle

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

DIR = "training_artefacts/"
print("Creating directory for training artefacts")

if not os.path.exists(DIR):
    os.makedirs(DIR)

path = os.path.join("../data", "dataset_26_nursery.arff")
data = loadarff(path)
data = pd.DataFrame(data[0])
data = data.select_dtypes([object])
data = data.stack().str.decode("utf-8").unstack()

print()

target_encoder = LabelEncoder()
target_vals = target_encoder.fit_transform(data["class"].values)
target_dataframe = pd.DataFrame({"class": target_vals})
data = data.drop(columns=["class"], inplace=False)

feature_encoder = OneHotEncoder()
x_encoded = feature_encoder.fit_transform(data).toarray()
feature_dataframe = pd.DataFrame(
    x_encoded, columns=feature_encoder.get_feature_names_out()
)

trainX, testX, trainy, testy = train_test_split(
    feature_dataframe.values,
    target_dataframe.values.flatten(),
    test_size=0.7,
    random_state=42,
)

print(f"Saving training/testing data to ./{DIR}")
np.savetxt(DIR + "trainX.txt", trainX, fmt="%d")
np.savetxt(DIR + "trainy.txt", trainy, fmt="%d")
np.savetxt(DIR + "testX.txt", testX, fmt="%d")
np.savetxt(DIR + "testy.txt", testy, fmt="%d")

# These hyperparameters lead to a dangerously disclosive trained model
DISCLOSIVE_PARAMS = {}
DISCLOSIVE_PARAMS["min_samples_split"] = 2
DISCLOSIVE_PARAMS["min_samples_leaf"] = 1
DISCLOSIVE_PARAMS["max_depth"] = None
DISCLOSIVE_PARAMS["bootstrap"] = False

print("Training disclosive model with the following hyperparameters:")
print(str(DISCLOSIVE_PARAMS))

target_model = RandomForestClassifier(**DISCLOSIVE_PARAMS)
target_model.fit(trainX, trainy)

train_acc = accuracy_score(trainy, target_model.predict(trainX))
test_acc = accuracy_score(testy, target_model.predict(testX))
print(f"Training accuracy on disclosive model: {train_acc:.2f}")
print(f"Testing accuracy on disclosive model: {test_acc:.2f}")

FILENAME = f"{DIR}disclosive_random_forest.sav"
print(f"Saving disclosive model to {FILENAME}")
with open(FILENAME, "wb") as fp:
    pickle.dump(target_model, fp)

SAFE_PARAMS = {}
SAFE_PARAMS["min_samples_split"] = 20
SAFE_PARAMS["min_samples_leaf"] = 10
SAFE_PARAMS["max_depth"] = 5
SAFE_PARAMS["bootstrap"] = True

print()
print(f"Training safer model with the following hyperparameters: ({SAFE_PARAMS})")

target_model = RandomForestClassifier(**SAFE_PARAMS)
target_model.fit(trainX, trainy)

train_acc = accuracy_score(trainy, target_model.predict(trainX))
test_acc = accuracy_score(testy, target_model.predict(testX))

print(f"Training accuracy on safe model: {train_acc:.2f}")
print(f"Testing accuracy on safe model: {test_acc:.2f}")

FILENAME = f"{DIR}safe_random_forest.sav"
print(f"Saving safe model to {FILENAME}")
with open(FILENAME, "wb") as fp:
    pickle.dump(target_model, fp)
