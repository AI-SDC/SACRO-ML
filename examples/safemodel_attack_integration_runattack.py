""" workimg on how to integrate attacks into safemosdel classes
Invoke this code from the root AI-SDC folder with
python -m examples.test_sagfemodel_attack_integration

"""

import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


from safemodel.classifiers import SafeDecisionTreeClassifier
from attacks.dataset import Data


if __name__ == "__main__":


    # [Researcher] Access a dataset
    nursery_data = fetch_openml(data_id=26, as_frame=True)
    x = np.asarray(nursery_data.data, dtype=str)
    y = np.asarray(nursery_data.target, dtype=str)
<<<<<<< HEAD
        #relabel tiny class
    for i in range (y.shape[0]):
        if y[i] == 'recommend':
            y[i]='not_recom'
        
=======
    #cast to a binary problem
    for i in range (y.shape[0]):
        if y[i]=='not_recom':
            y[i]= 0
        else :
            y[i]=1

>>>>>>> d5f3c966ea5ca5ef6cc463c08aaa966ff3bd0fd8
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
        test_size=0.5,
        stratify=y,
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

    # [TRE / Researcher] Wrap the data in a dataset object
    the_data = Data()
    the_data.name = "nursery"
    the_data.add_processed_data(x_train, y_train, x_test, y_test)
    the_data.add_raw_data(x, y, x_train_orig, y_train_orig, x_test_orig, y_test_orig)
    for i in range(n_features):
        the_data.add_feature(nursery_data.feature_names[i], indices[i], "onehot")

    print(f"Dataset: {the_data.name}")
    print(f"Features: {the_data.features}")
    print(f"x_train shape = {np.shape(the_data.x_train)}")
    print(f"y_train shape = {np.shape(the_data.y_train)}")
    print(f"x_test shape = {np.shape(the_data.x_test)}")
    print(f"y_test shape = {np.shape(the_data.y_test)}")


    #build a model
    model = SafeDecisionTreeClassifier(random_state=1)
    model.fit(x_train, y_train)
    msg, disclosive = model.preliminary_check()



    ##check direct method
    print('==========> first running direct attacks')
    for attack_name in ['worst_case','attribute','lira']:
        print(f'===> running {attack_name} attack directly')
        fname=f"modelDOTrun_attack_output_{attack_name}"
        metadata= model.run_attack(the_data,attack_name,fname)
        print('metadata is:')
        for key,val in metadata.items():
            if  isinstance(val,dict):
                print(f" {key}")
                for key1,val2 in val.items():
                    print(f'   {key1} : {val2}')
            else:
                print(f' {key} : {val}')
