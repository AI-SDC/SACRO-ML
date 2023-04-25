# %%
# Sort out the path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# %% Get the nursery data
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from aisdc.preprocessing.loaders import get_data_sklearn
from aisdc.attacks.dataset import Data
from aisdc.attacks.worst_case_attack import WorstCaseAttack, WorstCaseAttackArgs


# Get a dataset
X, y = get_data_sklearn('nursery')
# Train / test split
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.95)
# train a model
rf = RandomForestClassifier(min_samples_leaf=2, min_samples_split=2, max_depth=None)
rf.fit(trainX, trainy)

train_acc = accuracy_score(trainy, rf.predict(trainX))
test_acc = accuracy_score(testy, rf.predict(testX))
print(f'Train acc: {train_acc:.2f}, Test acc: {test_acc:.2f}')

train_probs = rf.predict_proba(trainX)
test_probs = rf.predict_proba(testX)

# Make data object
nursery = Data()
nursery.add_processed_data(trainX, trainy, testX, testy)

# Run attacks
args = WorstCaseAttackArgs(n_dummy_reps=0, report_name=None)
wca = WorstCaseAttack(args)
wca.attack(nursery, rf)

# Make the output -- crash here!
output = wca.make_report()

# %%
import json
with open('example.json', 'r') as f:
    desired_format = json.loads(f.read())

# %%
# Convert output
def convert_output(dictionary_in):
    # TODO add counters
    # convert to the desired format -- temporary
    dict_out = {
        'log_id': '12345',
        'timestamp': 'right now',
        'attack_type': 'WorstCaseAttack',
        'attack_type_version': 'current',
        'model': 'RandomForestClassifier',
        'model_params': ['Something'], # q is this attack model params or target model params?
        'attack_instance_logger': {}
    }
    counter = 1
    for metric_set in output['attack_metrics']:
        new_key = f'instance_{counter}'
        counter += 1
        dict_out['attack_instance_logger'][new_key] = {'metrics_array': {k: v for k, v in metric_set.items()}}
    return dict_out

# %%
reformat_output = convert_output(output)
# %%
class AnalysisModule(object):
    def process_dict(self, input_dict: dict):
        raise NotImplementedError()
    def __str__(self):
        raise NotImplementedError()



class SummariseUnivariateMetricsModule(AnalysisModule):
    '''
    Module that summarises a set of chosen univariate metrics from the output dictionary
    '''
    def __init__(self, metric_list=None):
        if metric_list is None:
            metric_list = ['AUC', 'ACC', 'FDIF01']
        self.metric_list = metric_list
    
    def process_dict(self, input_dict: dict) -> dict:
        metric_dict = {m: [] for m in self.metric_list}
        for _, iteration_value in input_dict['attack_instance_logger'].items():
            for m in metric_dict:
                metric_dict[m].append(iteration_value['metrics_array'][m])
        output = {}
        for m in self.metric_list:
            output[m] = {
                'min': min(metric_dict[m]),
                'max': max(metric_dict[m]),
                'mean': np.mean(metric_dict[m]),
                'median': np.median(metric_dict[m])
            }
        return output

    def __str__(self):
        return "SummariseUnivariateMetricsModule"

s = SummariseUnivariateMetricsModule()
print(s.process_dict(reformat_output))

class SummariseAUCPvalsModule(object):
    def __init__(self, p_thresh: float=0.05, correction: str='bh'):
        self.p_thresh = p_thresh
        self.correction = correction

    def n_sig(self, p_val_list: list[float], correction: str='none') -> int:
        '''Compute the number of significant p-vals in a list with different corrections for
        multiple testing'''
        if correction == 'none':
            return len(np.where(np.array(p_val_list) <= self.p_thresh)[0])
        elif correction == 'bh':
            sorted_p = sorted(p_val_list)
            m = len(p_val_list)
            alpha = self.p_thresh
            comparators = np.arange(1, m + 1, 1) * alpha / m
            return (sorted_p <= comparators).sum()
        elif correction == 'bo': #bonferroni
            return len(np.where(np.array(p_val_list) <= self.p_thresh / len(p_val_list))[0])
        else:
            raise NotImplementedError() # need any others?
    
    def get_metric_list(self, input_dict: dict) -> list[float]:
        metric_list = []
        for _, iteration_value in input_dict['attack_instance_logger'].items():
            metric_list.append(iteration_value['metrics_array']['P_HIGHER_AUC'])
        return metric_list


    def process_dict(self, input_dict: dict) -> dict:
        '''Process the dict to summarise the number of significant AUC p-values'''
        p_val_list = self.get_metric_list(input_dict)
        output = {
            'n_total': len(p_val_list),
            'p_thresh': self.p_thresh,
            'n_sig_uncorrected': self.n_sig(p_val_list),
            'correction': self.correction,
            'n_sig_corrected': self.n_sig(p_val_list, self.correction)
        }
        return output
    
    def __str__(self):
        return f"SummariseAUCPvalsModule ({self.p_thresh})"



s = SummariseAUCPvalsModule(correction='bh')
print(s.process_dict(reformat_output))

class SummariseFDIFPvalsModule(SummariseAUCPvalsModule):
    '''Summarise the number of significant FDIF p-values'''
    # TODO do we want to parameterise which FDIF (01, 001, etc)?
    def get_metric_list(self, input_dict: dict) -> list[float]:
        metric_list = []
        for _, iteration_value in input_dict['attack_instance_logger'].items():
            metric_list.append(iteration_value['metrics_array']['PDIF01'])
        metric_list = [np.exp(-m) for m in metric_list]
        return metric_list
    def __str__(self):
        return f"SummariseFDIFPvalsModule ({self.p_thresh})"
    
s = SummariseFDIFPvalsModule(correction='bh')
print(s.process_dict(reformat_output))

# %%
# This is how it could be used neatly
modules = [
    SummariseUnivariateMetricsModule(),
    SummariseAUCPvalsModule(p_thresh=0.05),
    SummariseAUCPvalsModule(p_thresh=0.1),
    SummariseFDIFPvalsModule()
]

output = {str(m): m.process_dict(reformat_output) for m in modules}


# %%
