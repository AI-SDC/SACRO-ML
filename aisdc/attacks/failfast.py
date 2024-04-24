"""Failfast.py - class to evaluate metric for fail fast option."""

from __future__ import annotations

from typing import Any


class FailFast:
    """Class to check attack being successful or not for a given metric
    Note: An object of a FailFast is stateful and instance members
    (success_count and fail_count) will preserve values
    across repetitions for a test. For the new test
    a new object will require to be instantiated.
    """

    def __init__(self, attack_obj: Any):
        self.metric_name = attack_obj.attack_metric_success_name
        self.metric_success_thresh = attack_obj.attack_metric_success_thresh
        self.comp_type = attack_obj.attack_metric_success_comp_type
        self.success_count = 0
        self.fail_count = 0

    def check_attack_success(self, metric_dict: dict) -> bool:
        """A function to check if attack was successful for a given metric.

        Parameters
        ----------
        metric_dict : dict
            a dictionary with all computed metric values

        Returns
        -------
        success_status : bool
            a boolean value is returned based on the comparison for a given threshold

        Notes
        -----
        If value of a given metric value has a value meeting the threshold based on
        the comparison type returns true otherwise it returns false. This function
        also counts how many times the attack was successful (i.e. true) and
        how many times it was not successful (i.e. false).
        """
        metric_value = metric_dict[self.metric_name]
        success_status = False
        if self.comp_type == "lt":
            success_status = bool(metric_value < self.metric_success_thresh)
        elif self.comp_type == "lte":
            success_status = bool(metric_value <= self.metric_success_thresh)
        elif self.comp_type == "gt":
            success_status = bool(metric_value > self.metric_success_thresh)
        elif self.comp_type == "gte":
            success_status = bool(metric_value >= self.metric_success_thresh)
        elif self.comp_type == "eq":
            success_status = bool(metric_value == self.metric_success_thresh)
        elif self.comp_type == "not_eq":
            success_status = bool(metric_value != self.metric_success_thresh)

        if success_status:
            self._increment_success_count()
        else:
            self._incremenet_fail_count()

        return success_status

    def _increment_success_count(self) -> int:
        self.success_count += 1

    def _incremenet_fail_count(self) -> int:
        self.fail_count += 1

    def get_success_count(self) -> int:
        """Returns a count of attack being successful."""
        return self.success_count

    def get_fail_count(self):
        """Returns a count of attack being not successful."""
        return self.fail_count

    def get_attack_summary(self) -> dict:
        """Returns a dictionary of counts of attack being successful and not successful."""
        summary = {}
        summary["success_count"] = self.success_count
        summary["fail_count"] = self.fail_count
        return summary

    def check_overall_attack_success(self, attack_obj: Any) -> bool:
        """Returns true if the attack is successful for a given success count threshold."""
        overall_success_status = False
        if self.success_count >= attack_obj.attack_metric_success_count_thresh:
            overall_success_status = True
        return overall_success_status
