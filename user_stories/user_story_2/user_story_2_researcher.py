"""
User story 2 (best case) as researcher.

Details can be found here:
https://github.com/AI-SDC/AI-SDC/issues/141

Running
-------

Invoke this code from the root AI-SDC folder with
python -m example_notebooks.user_stories.user_story_2.user_story_2_researcher
"""

import logging
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from aisdc.attacks.target import Target  # pylint: disable=import-error
from aisdc.safemodel.classifiers import (  # pylint: disable=import-error
    SafeDecisionTreeClassifier,
)

from data_processing_researcher import process_dataset

def main():  # pylint: disable=too-many-locals
    """Create and train a model to be released."""
    directory = "training_artefacts"
    print("Creating directory for training artefacts")

    if not os.path.exists(directory):
        os.makedirs(directory)

    print()
    print("Acting as researcher...")
    print()
    filename = os.path.join("..", "user_stories_resources", "dataset_26_nursery.csv")
    print("Reading data from " + filename)
    data = pd.read_csv(filename)

    print()

    returned = process_dataset(data)

    x_transformed = returned['x_transformed']
    y_transformed = returned['y_transformed']

    n_features = np.shape(x_transformed)[1]

    row_indices = np.arange(np.shape(x_transformed)[0])

    # [Researcher] Split into training and test sets
    # target model train / test split - these are strings
    (
        x_train,
        x_test,
        y_train,
        y_test,
        indices_train,
        indices_test,
    ) = train_test_split(
        x_transformed,
        y_transformed,
        row_indices,
        test_size=0.5,
        stratify=y_transformed,
        shuffle=True,
    )

    indices = returned['indices']

    logging.getLogger("attack-reps").setLevel(logging.WARNING)
    logging.getLogger("prep-attack-data").setLevel(logging.WARNING)
    logging.getLogger("attack-from-preds").setLevel(logging.WARNING)

    # Build a model and request its release
    model = SafeDecisionTreeClassifier(random_state=1)
    model.fit(x_train, y_train)
    model.request_release(path=directory, ext="pkl")

    # Wrap the model and data in a Target object
    target = Target(model=model)
    target.name = "nursery"
    target.add_processed_data(x_train, y_train, x_test, y_test)
    target.add_raw_data(x_transformed, y_transformed)
    for i in range(n_features):
        target.add_feature(data.columns[i], indices[i], "onehot")

    # NOTE: we assume here that the researcher does not use the target.save() function
    # and instead provides only the model and the list of indices
    # which have been used to split the dataset

    print("Saving training/testing indices to " + directory)
    np.savetxt(os.path.join(directory, "indices_train.txt"), indices_train, fmt="%d")
    np.savetxt(os.path.join(directory, "indices_test.txt"), indices_test, fmt="%d")

    logging.info("Dataset: %s", target.name)
    logging.info("Features: %s", target.features)
    logging.info("x_train shape = %s", np.shape(target.x_train))
    logging.info("y_train shape = %s", np.shape(target.y_train))
    logging.info("x_test shape = %s", np.shape(target.x_test))
    logging.info("y_test shape = %s", np.shape(target.y_test))


if __name__ == "__main__":
    main()
