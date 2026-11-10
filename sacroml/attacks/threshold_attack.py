"""Run a simple threshold attack."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
from fpdf import FPDF
from sklearn.metrics import roc_auc_score, roc_curve

from sacroml.attacks import report
from sacroml.attacks.attack import Attack
from sacroml.attacks.model import ModelTask
from sacroml.attacks.signal import get_signals
from sacroml.attacks.target import Target
from sacroml.attacks.utils import tnr_at_fnr, tpr_at_fpr
from sacroml.metrics import min_max_disc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DP: int = 4  # display decimal places


class ThresholdAttack(Attack):
    """Threshold attack."""

    def __init__(
        self,
        output_dir: str = "outputs",
        write_report: bool = True,
        signal: str = "default",
    ) -> None:
        """Construct an object to execute a worst case attack.

        Parameters
        ----------
        output_dir : str
            Name of the directory where outputs are stored.
        write_report : bool
            Whether to generate a JSON and PDF report.
        signal : str
            Signal name, or ``"default"`` for the task default (``"logit"`` for
            classification, ``"log_mse"`` for regression).
        """
        super().__init__(output_dir=output_dir, write_report=write_report)
        self.signal: str = signal

    def __str__(self) -> str:
        """Return name of attack."""
        return "Threshold attack"

    @classmethod
    def attackable(cls, target: Target) -> bool:  # pragma: no cover
        """Return whether a target can be assessed with ThresholdAttack."""
        if (
            target.has_model()
            and target.has_data()
            and target.model.model_task in (ModelTask.REGRESSOR, ModelTask.CLASSIFIER)
        ) or target.has_probas():
            return True
        logger.info("WARNING: ThresholdAttack requires more Target details.")
        return False

    def _attack(self, target: Target) -> dict:
        """Run threshold attack.

        Parameters
        ----------
        target : attacks.target.Target
            target as a Target class object

        Returns
        -------
        dict
            Attack report.
        """
        # compute target model signals if possible
        if target.has_model() and target.has_data():
            members = get_signals(
                model=target.model,
                X=target.X_train,
                y=target.y_train,
                signal=self.signal,
            )
            nonmembers = get_signals(
                model=target.model, X=target.X_test, y=target.y_test, signal=self.signal
            )
        # use supplied target model probas if unable to compute
        else:
            members = target.proba_train
            nonmembers = target.proba_test

        # construct empty metadata
        self._construct_metadata()

        # execute attack
        self._run(members, nonmembers)

        # create the report
        output: dict[str, Any] = self._make_report()

        # write the report
        self._write_report(output)

        # return the report
        return output

    def _run(self, members: np.ndarray, nonmembers: np.ndarray) -> None:
        """Run a threshold attack.

        Parameters
        ----------
        members : np.ndarray
            Signals from member samples.
        nonmembers : np.ndarray
            Signals from non-member samples.
        """
        # Member signals should be higher than non-members
        if self.signal in ("logit", "log_mse", "default"):
            members = -members
            nonmembers = -nonmembers

        y_score = -np.hstack((members, nonmembers))
        y_truth = np.hstack((np.ones(len(members)), np.zeros(len(nonmembers))))

        fpr, tpr, roc_thresh = roc_curve(y_truth, y_score)
        auc = roc_auc_score(y_truth, y_score)

        self.attack_metrics = {
            "fpr": fpr,
            "tpr": tpr,
            "roc_thresh": roc_thresh,
        }

        global_metrics: dict = {
            "AUC": round(auc, DP),
        }

        values: tuple[float] = (0.0001, 0.001, 0.01, 0.1, 1, 5)
        for val in values:
            name = f"TPR @ {val:g}% FPR"
            global_metrics[name] = round(tpr_at_fpr(val, fpr, tpr), DP)

        for val in values:
            name = f"TNR @ {val:g}% FNR"
            global_metrics[name] = round(tnr_at_fnr(val, members, nonmembers), DP)

        for val in (0.01, 0.1, 0.2):
            fmax, fmin, fdif, pdif = min_max_disc(y_truth, y_score, x_prop=val)
            val_str = str(val).replace(".", "")
            global_metrics[f"FMAX{val_str}"] = round(fmax, DP)
            global_metrics[f"FMIN{val_str}"] = round(fmin, DP)
            global_metrics[f"FDIF{val_str}"] = round(fdif, DP)
            global_metrics[f"PDIF{val_str}"] = round(-pdif, DP)

        self.metadata["global_metrics"] = global_metrics

    def _make_report(self) -> dict[str, Any]:
        """Create attack report."""
        logger.info("Generating report")

        output: dict[str, Any] = {
            "log_id": self._instance_id,
            "log_time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "metadata": self.metadata,
            "attack_experiment_logger": self._get_attack_metrics_instances(),
        }
        return output

    def _make_pdf(self, output: dict) -> FPDF:
        """Create PDF report."""
        return report.create_threshold_report(output)

    def _get_attack_metrics_instances(self) -> dict:
        """Construct the metadata object after attacks."""
        return {"attack_instance_logger": {"instance_0": self.attack_metrics}}
