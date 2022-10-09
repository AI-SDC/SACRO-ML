"""
Tests attacks called via safemodel classes
uses a saubsmapled nursery dataset as this tests more of the attack code
currently using random forests
"""
import os

import pickle
from pathlib import Path
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from safemodel.classifiers import SafeRandomForestClassifier
from attacks.dataset import Data

# pylint: disable=too-many-locals,bare-except,duplicate-code

def cleanup_file(name:str):
    """removes unwanted files or directory"""
    if os.path.exists(name) and os.path.isfile(name):  # h5
        os.remove(name)


def get_nursery_dataset()->Data:
    """ returns a randomly sampled 10+10% of
    the nursery data set as a Data object
    if needed fetches it from openml and saves. it

    """

    the_file='tests/datasets/nursery_as_dataset.pickle'
    save_locally= True
    need_download=True

    if Path(the_file).is_file():
        try:
            with open(the_file,"rb") as f:
                the_data = pickle.load(f)
            need_download=False
        except:
            need_download = True

    if need_download:

        nursery_data = fetch_openml(data_id=26, as_frame=True)
        x = np.asarray(nursery_data.data, dtype=str)
        y = np.asarray(nursery_data.target, dtype=str)
        #change labels from recommend to priority for the two odd cases
        num=len(y)
        for i in range (num):
            if y[i]== 'recommend':
                y[i]='priority'

        n_features = np.shape(x)[1]
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



        # [Researcher] Split into training and test sets
        # target model train / test split - these are strings
        (x_train_orig, x_test_orig, y_train_orig, y_test_orig,) = train_test_split(
            x,
            y,
            test_size=0.2,
            stratify=y,
            shuffle=True,
        )
        print(f'train distribution: {np.unique(y_train_orig,return_counts=True)}\n'
              f'test distribution: {np.unique(y_test_orig,return_counts=True)}\n'
             )




        #now resample the training data reduce number of examples
        _,x_train_orig,_, y_train_orig = train_test_split(
                                            x_train_orig,
                                            y_train_orig,
                                            test_size=0.25,
                                            stratify=y_train_orig,
                                            shuffle=True,
                                            )
        print(f'train distribution: {np.unique(y_train_orig,return_counts=True)}\n'
              f'test distribution: {np.unique(y_test_orig,return_counts=True)}\n'
             )



        # [Researcher] Preprocess dataset
        # one-hot encoding of features and integer encoding of labels
        label_enc = LabelEncoder()
        feature_enc = OneHotEncoder()
        x_train = feature_enc.fit_transform(x_train_orig).toarray()
        y_train = label_enc.fit_transform(y_train_orig)
        x_test = feature_enc.transform(x_test_orig).toarray()
        y_test = label_enc.transform(y_test_orig)

        # [TRE / Researcher] Wrap the data in a dataset object
        the_data = Data()
        the_data.name = "nursery"
        the_data.add_processed_data(x_train, y_train, x_test, y_test)
        the_data.add_raw_data(x, y,
                              x_train_orig, y_train_orig,
                              x_test_orig, y_test_orig)
        for i in range(n_features):
            the_data.add_feature(nursery_data.feature_names[i],
                                 indices[i], "onehot")

    if need_download and save_locally:
        #make directory if needed then save
        output_file = Path(the_file)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        with open(the_file, 'wb') as f:
            pickle.dump(the_data, f)

    return the_data

def test_run_attack_lira():
    """ calls the lira attack via safemodel"""
    the_data = get_nursery_dataset()

    #build a model
    model = SafeRandomForestClassifier(random_state=1)
    model.fit(the_data.x_train, the_data.y_train)
    _, disclosive = model.preliminary_check()
    assert not disclosive

    print(np.unique(the_data.y_test, return_counts=True))

    print(np.unique(model.predict(the_data.x_test),return_counts=True))

    fname="delete-me"
    metadata= model.run_attack(the_data,"lira-n10",fname)
    cleanup_file('delete-me.json')
    assert len(metadata) >0 #something has been added



def test_run_attack_worstcase():
    """ calls the worst case attack via safemodel"""
    the_data = get_nursery_dataset()

    #build a model
    model = SafeRandomForestClassifier(random_state=1)
    model.fit(the_data.x_train, the_data.y_train)
    _, disclosive = model.preliminary_check()
    assert not disclosive

    fname="delete-me"
    metadata= model.run_attack(the_data,"worst_case",fname)
    cleanup_file("delete-me.json")
    #cleanup_files("modelDOTrun_attack_output_worst_case.json")
    assert len(metadata) >0 #something has been added

def test_run_attack_attribute():
    """ calls the attribute  attack via safemodel"""
    the_data = get_nursery_dataset()

    #build a model
    model = SafeRandomForestClassifier(random_state=1)
    model.fit(the_data.x_train, the_data.y_train)
    _, disclosive = model.preliminary_check()
    assert not disclosive

    fname="delete-me"
    metadata= model.run_attack(the_data,"attribute",fname)
    for outfile in ['delete-me.json',
                    'aia_example.json',
                    'aia_example.pdf',
                    'aia_report_cat_frac.png',
                    'aia_report_cat_risk.png']:
        cleanup_file(outfile)
    assert len(metadata) >0 #something has been added
