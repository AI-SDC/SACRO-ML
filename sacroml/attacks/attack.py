"""Base class for an attack object."""

from __future__ import annotations

import inspect
import logging
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import numpy as np
from fpdf import FPDF

from sacroml.attacks import report
from sacroml.attacks.target import Target
from sacroml.version import __version__

logger = logging.getLogger(__name__)


class Attack(ABC):
    """Abstract Base class to represent an attack."""

    # Keys stripped from JSON output for every attack. Large arrays
    # (fpr/tpr/roc_thresh) bloat reports without adding info beyond AUC; the
    # `individual` block is per-record data that we externalise to a compressed
    # .npz file (see `_individual_npz_prefix`).
    _json_exclude_keys: frozenset[str] = frozenset(
        {"fpr", "tpr", "roc_thresh", "individual"}
    )

    # Subclasses opt in to externalising per-record `individual` data to a
    # compressed .npz file by setting a non-empty prefix (e.g. "lira").
    _individual_npz_prefix: str = ""

    def __init__(self, output_dir: str = "outputs", write_report: bool = True) -> None:
        """Instantiate an attack.

        Parameters
        ----------
        output_dir : str
            name of the directory where outputs are stored
        write_report : bool
            Whether to generate a JSON and PDF report.
        """
        self.output_dir: str = output_dir
        self.write_report: bool = write_report
        self.attack_metrics: dict | list = {}
        self.metadata: dict = {}
        # Stable per-instance ID used as both the JSON `log_id` and a suffix on
        # any externalised .npz files, so two runs in the same output_dir don't
        # clobber each other and the artefacts stay linked.
        self._instance_id: str = str(uuid.uuid4())
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Create folder for saving trained shadow models
        self.shadow_path: str = os.path.normpath(f"{self.output_dir}/shadow_models")
        os.makedirs(self.shadow_path, exist_ok=True)

    @classmethod
    @abstractmethod
    def attackable(cls, target: Target) -> bool:
        """Return whether a given target can be assessed with an attack."""

    @abstractmethod
    def _attack(self, target: Target) -> dict:
        """Run an attack."""

    def attack(self, target: Target) -> dict:
        """Check whether an attack can be performed and run the attack."""
        return self._attack(target) if type(self).attackable(target) else {}

    def _construct_metadata(self) -> None:
        """Generate attack metadata."""
        self.metadata = {
            "sacroml_version": __version__,
            "attack_name": str(self),
            "attack_params": self.get_params(),
            "global_metrics": {},
        }

    @abstractmethod
    def _get_attack_metrics_instances(self) -> dict:
        """Get metrics for each individual repetition of an attack."""

    @abstractmethod
    def _make_pdf(self, output: dict) -> FPDF | None:
        """Create PDF report."""

    def _make_report(self, target: Target) -> dict[str, Any]:
        """Create attack report."""
        logger.info("Generating report")
        self._construct_metadata()

        if target.model is not None:
            self.metadata["target_model"] = target.model.model_name
            self.metadata["target_model_params"] = target.model.model_params
            self.metadata["target_train_params"] = target.model.train_params

        output: dict[str, Any] = {
            "log_id": self._instance_id,
            "log_time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "metadata": self.metadata,
            "attack_experiment_logger": self._get_attack_metrics_instances(),
        }
        return output

    def _individual_extras(self, instance_key: str) -> dict[str, np.ndarray]:
        """Return extra arrays to include in the .npz alongside `individual`.

        Override in subclasses that need to persist arrays not part of the
        per-record `individual` block (e.g. LiRA's `y_pred_proba`/`y_test`
        used for ROC recomputation).
        """
        del instance_key
        return {}

    def _externalise_individual(self, output: dict) -> None:
        """Write per-record `individual` blocks to compressed .npz files.

        For each instance under `attack_experiment_logger.attack_instance_logger`
        that contains an `individual` key, write its contents (plus any extras
        from `_individual_extras`) to a compressed .npz and add an
        `individual_file` pointer on the instance. The original `individual`
        block stays in memory (used by PDF generation and meta-attack); it is
        stripped from the JSON via `_json_exclude_keys`.
        """
        if not self._individual_npz_prefix:
            return
        instances = output.get("attack_experiment_logger", {}).get(
            "attack_instance_logger", {}
        )
        for key, inst in instances.items():
            individual = inst.get("individual")
            if not individual:
                continue
            extras = self._individual_extras(key)
            arrays: dict[str, np.ndarray] = {
                k: np.asarray(v) for k, v in individual.items()
            }
            arrays.update({k: np.asarray(v) for k, v in extras.items()})
            fname = (
                f"{self._individual_npz_prefix}_individual_"
                f"{self._instance_id[:8]}_{key}.npz"
            )
            np.savez_compressed(os.path.join(self.output_dir, fname), **arrays)
            inst["individual_file"] = fname

    def _write_report(self, output: dict) -> None:
        """Write report as JSON and PDF."""
        dest: str = os.path.join(self.output_dir, "report")
        if self.write_report:
            logger.info("Writing report: %s.json %s.pdf", dest, dest)
            self._externalise_individual(output)
            report.write_json(output, dest, exclude_keys=self._json_exclude_keys)
            pdf_report: FPDF | None = self._make_pdf(output)
            if pdf_report is not None:
                report.write_pdf(dest, pdf_report)

    @abstractmethod
    def __str__(self) -> str:
        """Return the string representation of an attack."""

    @classmethod
    def _get_param_names(cls) -> list[str]:
        """Get parameter names."""
        init_signature: inspect.Signature = inspect.signature(cls.__init__)
        return [
            p.name
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]

    def get_params(self) -> dict:
        """Get parameters for this attack.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out: dict = {}
        for key in self._get_param_names():
            out[key] = getattr(self, key)
        return out
