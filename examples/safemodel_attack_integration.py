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


print(f'model.get_params(). returns {model.get_params()}')


the_data = Data()

the_data.add_processed_data(x_train,y_train,x_test,y_test)

ok=True
if(ok):
    ##check direct method
    print('==========> first running direct attacks')
    for attack_name in ['lira']:#,'worst_case','attribute']:
        print(f'running {attack_name} attack directly')
        metadata= model.run_attack(the_data,attack_name,"modelDOTrun_attack_output")
        print(f'metadata is {metadata}')

    print('==========> now running request_release()')
    model.request_release("test.sav",the_data)
