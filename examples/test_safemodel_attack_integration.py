

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split



from safemodel.classifiers import SafeDecisionTreeClassifier
from attacks.dataset import Data

"""Returns data for testing."""
cancer = datasets.load_breast_cancer()
x = np.asarray(cancer['data'], dtype=np.float64)
y = np.asarray(cancer['target'], dtype=np.float64)
x_train, x_test, y_train, y_test = train_test_split(
                                     x, y, test_size=0.33, random_state=42) 


model = SafeDecisionTreeClassifier(random_state=1)
model.fit(x_train, y_train)
msg, disclosive = model.preliminary_check()

the_data = Data()

the_data.add_processed_data(x_train,y_train,x_test,y_test)

model.run_attack(the_data,"worst_case","anyoldfile")

