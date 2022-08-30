""" workimg on how to integrate attacks into safemosdel classes
Invoke this code from the root AI-SDC folder with
python -m examples.test_sagfemodel_attack_integration

"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split



from safemodel.classifiers import SafeDecisionTreeClassifier
from attacks.dataset import Data


cancer = datasets.load_breast_cancer()
x = np.asarray(cancer['data'], dtype=np.float64)
y = np.asarray(cancer['target'], dtype=np.float64)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

model = SafeDecisionTreeClassifier(random_state=1)
model.fit(x_train, y_train)
msg, disclosive = model.preliminary_check()

the_data = Data()

the_data.add_processed_data(x_train,y_train,x_test,y_test)

direct=True
if(direct):
    ##check direct method
    print('==========> first running direct attacks')
    for attack_name in ['lira','worst_case','attribute']:
        print(f'===> running {attack_name} attack directly')
        fname=f"modelDOTrun_attack_output_{attack_name}"
        metadata= model.run_attack(the_data,attack_name,fname)
        print(f'metadata is:')
        for key,val in metadata.items():
            if  isinstance(val,dict):
                print(f" {key}")
                for key1,val2 in val.items():
                    print(f'   {key1} : {val2}')
            else:
                print(f' {key} : {val}')
