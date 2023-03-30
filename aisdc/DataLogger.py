import random
import datetime
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import json
from collections import Counter
import numpy as np
import sys
import pickle

class DataLogger:
    def __init__(self, model_filename=None, data_filename=None, attack_type=None, attack_type_version=None, fail_fast_threshold=None):

        self.log = {}

        self.log['log_id'] = random.randint(1,100)
        self.log['timestamp'] = str(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        self.log['model_filename'] = model_filename
        self.log['data_filename'] = data_filename
        self.log['attack_type'] = attack_type
        self.log['attack_type_version'] = attack_type_version
        self.log['fail_fast_threshold'] = fail_fast_threshold

        self.log['attack_instance_logger'] = {}

    def generate_log(self):
        log_json = json.dumps(self.log)
        return log_json

    def generate_metrics(self):
        keys = ['true_labels_distribution','predicted_labels_distribution','metrics']

        metrics_dict = {}
        for k in keys:
            metrics_dict[k] = self.log[k]

        return metrics_dict
    
    def log_metrics(self, y_true, y_pred):
        self.log['true_labels_distribution'] = str(Counter(y_true))
        self.log['predicted_labels_distribution'] = str(Counter(y_pred))

        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        metrics_dict = {}
        metrics_dict['AUC'] = (metrics.roc_auc_score(y_true, y_pred))
        metrics_dict["TPR"] = round(float(tp / (tp + fn)), 8)
        metrics_dict["FPR"] = round(float(fp / (fp + tn)), 8)
        metrics_dict["FNR"] = round(float(fn / (tp + fn)), 8)
        metrics_dict["TNR"] = round(float(tn / (tn + fp)), 8)
        metrics_dict['acc'] = (round(float((tp + tn) / (tp + fp + fn + tn)), 8))
        metrics_dict['confusion_matrix'] = [int(tn), int(fp), int(fn), int(tp)]

        self.log['metrics'] = metrics_dict
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
    model_filename = "rf_model.sav"
    data_filename = "data.csv"
    
    d = DataLogger(model_filename=model_filename, data_filename=data_filename, attack_type='worst_case',attack_type_version='not_supported')
    
    rf = pickle.load(open(model_filename,'rb'))
    d.log_model(rf)

    dummy_y_true = [1,0,0,1,0,1,1,0,1,0,0,0,1,1,0]
    for i in range(4):
        dummy_y_pred = np.random.choice([0,1], size=len(dummy_y_true))
        d.log_instance(dummy_y_true, dummy_y_pred)

    print(d.generate_log())
    