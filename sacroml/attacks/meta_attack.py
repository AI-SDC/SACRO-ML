"""Meta-attack: aggregate per-record vulnerability across multiple privacy attacks.

Runs multiple privacy attacks (LiRA, QMIA, Structural) on the same Target,
extracts per-record vulnerability scores from each, and aggregates them into
a unified pandas DataFrame with two-level aggregation:

  Level 1 — within-attack: mean, std, and consistency across repeated runs.
  Level 2 — cross-attack:  arithmetic/geometric mean of MIA scores,
            binary structural flag, and total vulnerability count.

Reference: AI-SDC/SACRO-ML#428
"""

from __future__ import annotations

import logging

import pandas as pd
from fpdf import FPDF

from sacroml.attacks.attack import Attack
from sacroml.attacks.target import Target

logger = logging.getLogger(__name__)


class MetaAttack(Attack):
    """Aggregate per-record vulnerability across multiple privacy attacks.

    Parameters
    ----------
    attacks : list[tuple]
        Each entry is ``(name, params)`` or ``(name, params, n_reps)``.
        *name* must be one of :pyattr:`SUPPORTED_ATTACKS`.
        *params* is a dict of keyword arguments forwarded to the sub-attack
        constructor.  *n_reps* (default 1) is the number of independent
        repetitions; useful for stochastic attacks like LiRA.
    mia_threshold : float
        Score above which a record is flagged as MIA-vulnerable.
    k_threshold : int or None
        k-anonymity value below which a record is structurally vulnerable.
        ``None`` reads the default from the ACRO risk-appetite config.
    output_dir : str
        Directory for all outputs (sub-attack subdirectories, report, CSV).
    write_report : bool
        Whether to write JSON report and CSV to disk.
    """

    SUPPORTED_ATTACKS: set[str] = {"lira", "qmia", "structural"}
    """Attacks that expose per-record vulnerability scores."""

    MIA_ATTACKS: set[str] = {"lira", "qmia"}
    """Subset of supported attacks that produce membership-inference scores."""

    def __init__(
        self,
        attacks: list[tuple],
        mia_threshold: float = 0.5,
        k_threshold: int | None = None,
        output_dir: str = "outputs",
        write_report: bool = True,
    ) -> None:
        super().__init__(output_dir=output_dir, write_report=write_report)

        self.attacks: list[tuple[str, dict, int]] = self._parse_attacks(attacks)
        self.mia_threshold: float = mia_threshold

        if k_threshold is None:
            from acro import ACRO

            self.k_threshold: int = ACRO("default").config["safe_threshold"]
        else:
            self.k_threshold = k_threshold

        self.vulnerability_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_attacks(attacks: list[tuple]) -> list[tuple[str, dict, int]]:
        """Normalise and validate the *attacks* specification.

        Accepts 2-tuples ``(name, params)`` — *n_reps* defaults to 1 — or
        3-tuples ``(name, params, n_reps)``.

        Raises
        ------
        ValueError
            If a tuple has the wrong length, if *name* is not in
            :pyattr:`SUPPORTED_ATTACKS`, or if *n_reps* is not a positive
            integer.
        """
        specs: list[tuple[str, dict, int]] = []
        for entry in attacks:
            if len(entry) == 2:
                name, params = entry
                n_reps = 1
            elif len(entry) == 3:
                name, params, n_reps = entry
            else:
                raise ValueError(
                    f"Expected (name, params) or (name, params, n_reps), "
                    f"got tuple of length {len(entry)}: {entry}"
                )

            if name not in MetaAttack.SUPPORTED_ATTACKS:
                raise ValueError(
                    f"Unsupported attack: '{name}'. MetaAttack requires "
                    f"per-record scores. Supported: "
                    f"{sorted(MetaAttack.SUPPORTED_ATTACKS)}"
                )

            if not isinstance(n_reps, int) or n_reps < 1:
                raise ValueError(
                    f"n_reps must be a positive integer, got {n_reps!r}"
                )

            specs.append((name, dict(params), n_reps))
        return specs

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    @classmethod
    def attackable(cls, target: Target) -> bool:
        """Return whether *target* can be assessed with the meta-attack."""
        return target.has_model() and target.has_data()

    def _attack(self, target: Target) -> dict:
        """Run all sub-attacks and aggregate per-record vulnerabilities."""
        raise NotImplementedError("Stage 2")  # implemented in next commit

    def _get_attack_metrics_instances(self) -> dict:
        """Return metrics in the standard report structure."""
        raise NotImplementedError("Stage 5")  # implemented in later commit

    def _make_pdf(self, output: dict) -> FPDF | None:
        """Return ``None`` — PDF generation is not yet implemented."""
        return None

    def __str__(self) -> str:
        return "Meta Attack"
