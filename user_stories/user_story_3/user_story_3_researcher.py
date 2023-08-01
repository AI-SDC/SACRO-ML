"""
User story 3 as researcher.

Details can be found here:
https://github.com/AI-SDC/AI-SDC/issues/141

Running
-------

Invoke this code from the root AI-SDC folder with
python -m example_notebooks.user_stories.user_story_3.user_story_3_researcher
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

print()
print("Acting as researcher...")
print()

directory = "training_artefacts"
print("Creating directory for training artefacts")

if not os.path.exists(directory):
    os.makedirs(directory)

filename = "../user_stories_resources/dataset_26_nursery.csv"
print("Reading data from " + filename)
data = pd.read_csv(filename)

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

print("Saving training/testing data to ./" + directory)
np.savetxt(os.path.join(directory, "trainX.txt"), trainX, fmt="%d")
np.savetxt(os.path.join(directory, "trainy.txt"), trainy, fmt="%d")
np.savetxt(os.path.join(directory, "testX.txt"), testX, fmt="%d")
np.savetxt(os.path.join(directory, "testy.txt"), testy, fmt="%d")

# These hyperparameters lead to a dangerously disclosive trained model
DISCLOSIVE_HYPERPARAMETERS = {}
DISCLOSIVE_HYPERPARAMETERS["min_samples_split"] = 2
DISCLOSIVE_HYPERPARAMETERS["min_samples_leaf"] = 1
DISCLOSIVE_HYPERPARAMETERS["max_depth"] = None
DISCLOSIVE_HYPERPARAMETERS["bootstrap"] = False

print(
    "Training disclosive model with the following hyperparameters: "
    + str(DISCLOSIVE_HYPERPARAMETERS)
)
target_model = RandomForestClassifier(**DISCLOSIVE_HYPERPARAMETERS)
target_model.fit(trainX, trainy)

train_acc = accuracy_score(trainy, target_model.predict(trainX))
test_acc = accuracy_score(testy, target_model.predict(testX))
print(f"Training accuracy on disclosive model: {train_acc:.2f}")
print(f"Testing accuracy on disclosive model: {test_acc:.2f}")

filename = os.path.join(directory, "disclosive_random_forest.sav")
print("Saving disclosive model to " + filename)
with open(filename, "wb") as f:
    pickle.dump(target_model, f)
