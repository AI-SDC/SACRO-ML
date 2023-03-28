import random
import datetime
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import json
from collections import Counter
import numpy as np
import sys

class DataLogger:
    def __init__(self, attack_type=None, attack_type_version=None):

        self.log = {}

        self.log['log_id'] = random.randint(1,100)
        self.log['timestamp'] = str(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        self.log['attack_type'] = attack_type
        self.log['attack_type_version'] = attack_type_version

        self.log['attack_instance_logger'] = {}

    def generate_log(self):
        log_json = json.dumps(self.log)
        return log_json

    def generate_metrics(self):
        keys = ['True labels distribution','Predicted labels distribution','metrics_array']

        metrics_dict = {}
        for k in keys:
            metrics_dict[k] = self.log[k]

        return metrics_dict
    
    def log_metrics(self, y_true, y_pred):
        self.log['True labels distribution'] = str(Counter(y_true))
        self.log['Predicted labels distribution'] = str(Counter(y_pred))

        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        metrics_list = []
        metrics_list.append(metrics.roc_auc_score(y_true, y_pred))
        metrics_list.append(round(float(tp / (tp + fn)), 8))
        metrics_list.append(round(float(fp / (fp + tn)), 8))
        metrics_list.append(round(float(tn / (tn + fp)), 8))
        metrics_list.append(round(float((tp + tn) / (tp + fp + fn + tn)), 8))

        self.log['metrics_array'] = metrics_list
        # will call metrics.py from within attacks

    def log_model(self,clf):
        # needs more thought
        self.log['model'] = clf.__class__.__name__
        self.log['model_params'] = clf.get_params()

    def log_instance(self,y_true,y_pred):
        d_ = DataLogger()
        d_.log_metrics(y_true, y_pred)

        log = d_.generate_metrics()

        instance_number = len(self.log['attack_instance_logger'])+1
        self.log['attack_instance_logger']['instance_'+str(instance_number)] = log

    def log_notes(self,note):
        self.log['notes'] = note


if __name__ == '__main__':
    print("Hello World!")
    
    d = DataLogger(attack_type='worst_case',attack_type_version='not_supported')
    
    rf = RandomForestClassifier()
    d.log_model(rf)

    dummy_y_true = [1,0,0,1,0,1,1,0,1,0,0,0,1,1,0]
    for i in range(10):
        dummy_y_pred = np.random.choice([0,1], size=len(dummy_y_true))
        d.log_instance(dummy_y_true, dummy_y_pred)

    print(d.generate_log())
    