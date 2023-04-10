"""failfast.py - class to evaluate metric for fail fast option"""

from __future__ import annotations

class FailFast:  # pylint: disable=too-many-instance-attributes
    def __init__(self, metric_name, metric_success_thresh, comp_type):
        self.metric_name = metric_name
        self.metric_success_thresh = metric_success_thresh
        self.comp_type = comp_type
        self.success_count = 0 
        self.fail_count = 0        

    def check_attack_success(self, metric_dict):        
        metric_value = metric_dict[self.metric_name]        
        success_status=False
        if self.comp_type == 'lt':
            if metric_value < self.metric_success_thresh:
                success_status = True 
                self.success_count += 1
            else:
                success_status = False
                self.fail_count += 1                   
        elif self.comp_type == 'lte':
            if metric_value <= self.metric_success_thresh:
                success_status = True 
                self.success_count += 1
            else:
                success_status = False
                self.fail_count += 1
        elif self.comp_type == 'gt':
            if metric_value > self.metric_success_thresh:
                success_status = True 
                self.success_count += 1
            else:
                success_status = False
                self.fail_count += 1
        elif self.comp_type == 'gte':
            if metric_value >= self.metric_success_thresh:
                success_status = True 
                self.success_count += 1
            else:
                success_status = False
                self.fail_count += 1
        elif self.comp_type == 'eq':
            if metric_value == self.metric_success_thresh:
                success_status = True 
                self.success_count += 1
            else:
                success_status = False
                self.fail_count += 1
        elif self.comp_type == 'not_eq':
            if metric_value != self.metric_success_thresh:
                success_status = True 
                self.success_count += 1
            else:
                success_status = False
                self.fail_count += 1
        return success_status           

    def get_success_count(self):
        return self.success_count

    def get_fail_count(self):
        return self.fail_count

    def get_attack_summary(self) -> dict:
        summary={}
        summary['success_count'] = self.success_count
        summary['fail_count'] = self.fail_count
        return summary



