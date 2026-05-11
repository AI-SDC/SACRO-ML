"""Meta-attack: aggregate per-record vulnerability across multiple privacy attacks.

Runs multiple privacy attacks (LiRA, QMIA, Structural) on the same Target,
extracts per-record vulnerability scores from each, and aggregates them into
a unified pandas DataFrame with two-level aggregation:

  Level 1 — within-attack: mean, std, and consistency across repeated runs.
  Level 2 — cross-attack:  arithmetic/geometric mean of MIA scores,
            binary structural flag, and total vulnerability count.

Supports three operating modes via the *behaviour* parameter:

  ``'run_all'`` (default)
      Run every specified attack from scratch.

  ``'use_existing_only'``
      Read per-record scores from existing ``report.json`` files in
      *report_dir*; no new attacks are executed.  Use when attacks were
      already run (possibly at great computational cost) and you only want
      to collate their results.

  ``'fill_missing'``
      Load any attacks already present in *report_dir* and run only those
      not yet found.  Saves redundant computation when some attacks have
      been run but others have not.

Reference: AI-SDC/SACRO-ML#428
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os

import numpy as np
import pandas as pd
from fpdf import FPDF

from sacroml import metrics
from sacroml.attacks.attack import Attack
from sacroml.attacks.target import Target

logger = logging.getLogger(__name__)


SUPPORTED_ATTACKS: set[str] = {"lira", "qmia", "structural"}
"""Attacks that expose per-record vulnerability scores."""

MIA_ATTACKS: set[str] = {"lira", "qmia"}
"""Subset of supported attacks that produce membership-inference scores."""

BEHAVIOUR_RUN_ALL: str = "run_all"
BEHAVIOUR_USE_EXISTING: str = "use_existing_only"
BEHAVIOUR_FILL_MISSING: str = "fill_missing"

# Maps the human-readable attack_name stored in report metadata → factory key.
# Keys must match the __str__() return value of each corresponding attack class.
# Values must be a subset of SUPPORTED_ATTACKS.
_REPORT_NAME_TO_KEY: dict[str, str] = {
    "LiRA Attack": "lira",
    "QMIA Attack": "qmia",
    "Structural Attack": "structural",
}

_MIA_SCORE_FIELDS: dict[str, str] = {
    "lira": "score",
    "qmia": "member_prob",
}
"""Maps factory key → field name inside ``attack_metrics[N]["individual"]``.

Used only by :meth:`MetaAttack._extract_mia_scores` (the live-attack path).
The disk-reading path (:meth:`MetaAttack._extract_scores_from_report`) uses
the same field names but looks them up directly rather than via this mapping.
"""

_EPS: float = 1e-10
"""Small constant to avoid log(0) in geometric mean computation."""


class MetaAttack(Attack):
    """Aggregate per-record vulnerability across multiple privacy attacks.

    Parameters
    ----------
    attacks : list[tuple]
        Each entry is ``(name, params)`` or ``(name, params, n_reps)``.
        *name* must be one of :data:`SUPPORTED_ATTACKS`.
        *params* is a dict of keyword arguments forwarded to the sub-attack
        constructor.  *n_reps* (default 1) is the number of independent
        repetitions; useful for stochastic attacks like LiRA.
    behaviour : str
        ``'run_all'`` (default), ``'use_existing_only'``, or
        ``'fill_missing'``.  See module docstring for details.
    report_dir : str or None
        Directory to scan for existing attack ``report.json`` files when
        *behaviour* is ``'use_existing_only'`` or ``'fill_missing'``.
        Defaults to *output_dir* when not provided.
    mia_threshold : float
        Score above which a record is flagged as MIA-vulnerable.
    k_threshold : int or None
        k-anonymity value below which a record is structurally vulnerable.
        ``None`` reads the default from the ACRO risk-appetite config.
    output_dir : str
        Directory for all outputs (sub-attack subdirectories, report, CSV).
    write_report : bool
        Whether to write JSON report and CSV to disk.
    keep_separate : bool
        Controls JSON output location.  ``False`` (default) appends the
        MetaAttack section to ``{report_dir}/report.json`` so it joins any
        sub-attack reports already there, matching the project convention.
        ``True`` writes a separate ``{output_dir}/report.json`` like the
        base class.  The CSV (``vulnerability_matrix.csv``) and PDF always
        follow the JSON output location.
    """

    def __init__(
        self,
        attacks: list[tuple | list],
        behaviour: str = "run_all",
        report_dir: str | None = None,
        mia_threshold: float = 0.5,
        k_threshold: int | None = None,
        output_dir: str = "outputs",
        write_report: bool = True,
        keep_separate: bool = False,
    ) -> None:
        super().__init__(output_dir=output_dir, write_report=write_report)
        # MetaAttack does not use shadow models; remove the empty directory
        # created by the base class so the output directory stays clean.
        with contextlib.suppress(OSError):
            os.rmdir(self.shadow_path)

        self.attacks: list[tuple[str, dict, int]] = self._parse_attacks(attacks)

        valid = {
            BEHAVIOUR_RUN_ALL,
            BEHAVIOUR_USE_EXISTING,
            BEHAVIOUR_FILL_MISSING,
        }
        if behaviour not in valid:
            raise ValueError(
                f"Unknown behaviour: {behaviour!r}. Expected one of {sorted(valid)}."
            )
        self.behaviour: str = behaviour
        self.report_dir: str = report_dir if report_dir is not None else output_dir
        self.keep_separate: bool = keep_separate

        self.mia_threshold: float = mia_threshold

        if k_threshold is None:
            from acro import ACRO  # noqa: PLC0415

            self.k_threshold: int = ACRO("default").config["safe_threshold"]
        else:
            self.k_threshold = k_threshold

        self.vulnerability_df: pd.DataFrame | None = None

        unknown = set(_REPORT_NAME_TO_KEY.values()) - SUPPORTED_ATTACKS
        if unknown:
            raise RuntimeError(
                f"_REPORT_NAME_TO_KEY references unsupported attacks: {unknown}. "
                "Update SUPPORTED_ATTACKS or fix the mapping."
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_attacks(attacks: list[tuple | list]) -> list[tuple[str, dict, int]]:
        """Normalise and validate the *attacks* specification.

        Accepts 2-tuples ``(name, params)`` — *n_reps* defaults to 1 — or
        3-tuples ``(name, params, n_reps)``.

        Raises
        ------
        ValueError
            If a tuple has the wrong length, if *name* is not in
            :data:`SUPPORTED_ATTACKS`, or if *n_reps* is not a positive
            integer.
        """
        if not attacks:
            raise ValueError("attacks must contain at least one entry.")

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
                    f"got entry of length {len(entry)}: {entry}"
                )

            if name not in SUPPORTED_ATTACKS:
                raise ValueError(
                    f"Unsupported attack: '{name}'. MetaAttack requires "
                    f"per-record scores. Supported: "
                    f"{sorted(SUPPORTED_ATTACKS)}"
                )

            if not isinstance(n_reps, int) or n_reps < 1:
                raise ValueError(f"n_reps must be a positive integer, got {n_reps!r}")

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
        """Run sub-attacks (or read existing) and aggregate per-record vulnerabilities.

        Behaviour is controlled by ``self.behaviour``:

        - ``'run_all'``: run every attack fresh.
        - ``'use_existing_only'``: scan *report_dir* for report.json files;
          extract scores without running any new attack.
        - ``'fill_missing'``: load existing results from *report_dir*,
          run only those not already present.

        Returns an empty dict ``{}`` when no scores are available — this can
        happen when no valid ``report.json`` files are found in
        ``'use_existing_only'`` mode, or when all sub-attacks fail.
        """
        # Step 1: Load existing results when not running entirely from scratch.
        existing_mia: dict[str, list[list[float]]] = {}
        existing_struct: dict[str, list[dict]] = {}

        if self.behaviour != BEHAVIOUR_RUN_ALL:
            existing_mia, existing_struct = self._scan_existing_reports()

        # Step 2: Populate score dicts — start from existing, then run new ones.
        mia_scores: dict[str, list[list[float]]] = dict(existing_mia)
        structural_scores: dict[str, list[dict]] = dict(existing_struct)

        if self.behaviour != BEHAVIOUR_USE_EXISTING:
            self._run_new_attacks(
                target, existing_mia, existing_struct, mia_scores, structural_scores
            )

        if not mia_scores and not structural_scores:
            logger.warning("No vulnerability scores collected; returning empty report.")
            return {}

        if target.X_train is None or target.X_test is None:
            logger.warning(
                "Target is missing X_train or X_test; returning empty report."
            )
            return {}

        n_train = len(target.X_train)
        n_test = len(target.X_test)
        self.vulnerability_df = self._build_dataframe(
            n_train, n_test, mia_scores, structural_scores
        )
        self._compute_global_metrics(n_train, n_test)

        output = self._make_report(target)
        self._write_report(output)
        return output

    # ------------------------------------------------------------------
    # Existing-report scanning
    # ------------------------------------------------------------------

    def _scan_existing_reports(
        self,
    ) -> tuple[dict[str, list[list[float]]], dict[str, list[dict]]]:
        """Scan *report_dir* for cached attack scores.

        Supports two on-disk layouts:

        1. **Canonical single-file layout**, ``{report_dir}/report.json``,
           where each individual attack has appended its own
           ``"AttackName_<uuid>"`` section via :class:`GenerateJSONModule`.
           This is the layout produced when LiRA, QMIA, and Structural are
           run separately with the same ``output_dir``.
        2. **Subdirectory-per-attack layout**, ``{report_dir}/<sub>/report.json``,
           where each sub-attack has its own ``report.json``.

        Both layouts are scanned, so a mixed setup also works.  The attack
        type is identified from the ``metadata.attack_name`` field; individual
        per-record scores are extracted from
        ``attack_experiment_logger["attack_instance_logger"]``.

        Returns
        -------
        tuple[dict, dict]
            ``(mia_scores, structural_scores)`` with the same structure used
            internally by :meth:`_attack`.
        """
        mia_scores: dict[str, list[list[float]]] = {}
        structural_scores: dict[str, list[dict]] = {}

        if not os.path.isdir(self.report_dir):
            logger.warning("report_dir %r does not exist.", self.report_dir)
            return mia_scores, structural_scores

        # Layout 1: top-level canonical report.json
        top_level = os.path.join(self.report_dir, "report.json")
        if os.path.isfile(top_level):
            self._extract_from_report_file(top_level, mia_scores, structural_scores)

        # Layout 2: subdirectory-per-attack
        try:
            entries = sorted(os.scandir(self.report_dir), key=lambda e: e.name)
        except OSError as exc:
            logger.warning(
                "Cannot scan report_dir %r: %s; skipping.", self.report_dir, exc
            )
            return mia_scores, structural_scores

        for entry in entries:
            if not entry.is_dir():
                continue
            sub_report = os.path.join(entry.path, "report.json")
            if not os.path.isfile(sub_report):
                continue
            self._extract_from_report_file(sub_report, mia_scores, structural_scores)

        return mia_scores, structural_scores

    def _extract_from_report_file(
        self,
        report_path: str,
        mia_scores: dict[str, list[list[float]]],
        structural_scores: dict[str, list[dict]],
    ) -> None:
        """Parse one ``report.json`` file, accumulating scores in place.

        Iterates every top-level ``"AttackName_<uuid>"`` section, identifies
        the attack via :data:`_REPORT_NAME_TO_KEY`, and extends the matching
        dict (``mia_scores`` or ``structural_scores``).  Unrecognised
        attack names are skipped with a debug log; unreadable files are
        skipped with a warning.
        """
        try:
            with open(report_path) as fh:
                report_data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read %s (%s); skipping.", report_path, exc)
            return

        for attack_data in report_data.values():
            if not isinstance(attack_data, dict):
                continue
            attack_name = attack_data.get("metadata", {}).get("attack_name", "")
            key = _REPORT_NAME_TO_KEY.get(attack_name)
            if key is None:
                logger.debug(
                    "Unrecognised attack_name %r in %s; skipping.",
                    attack_name,
                    report_path,
                )
                continue

            scores = self._extract_scores_from_report(attack_data, key)
            if scores is None:
                continue

            if key in MIA_ATTACKS:
                mia_scores.setdefault(key, []).extend(scores)  # type: ignore[arg-type]
            else:
                structural_scores.setdefault(key, []).extend(scores)  # type: ignore[arg-type]

            logger.info("Loaded existing %s results from %s.", key, report_path)

    def _extract_scores_from_report(  # noqa: C901
        self, report_data: dict, key: str
    ) -> list[list[float]] | list[dict] | None:
        """Extract per-record scores from a parsed report dict.

        Parameters
        ----------
        report_data : dict
            A single attack entry from a parsed ``report.json`` file — the dict
            value under one ``'AttackName_<uuid>'`` top-level key. Expected keys:
            ``'metadata'``, ``'attack_experiment_logger'``.
        key : str
            Factory key (``'lira'``, ``'qmia'``, or ``'structural'``).

        Returns
        -------
        list[list[float]] | list[dict] | None
            One entry per instance found in the report, in the format expected
            by :meth:`_build_dataframe`, or ``None`` when no individual scores
            are present.
        """
        try:
            logger_key = "attack_instance_logger"
            instances = report_data["attack_experiment_logger"][logger_key]
            if not isinstance(instances, dict):
                raise TypeError(f"Expected dict, got {type(instances).__name__}")
        except (KeyError, TypeError) as exc:
            logger.warning(
                "Unexpected report structure for %s (%s); skipping.", key, exc
            )
            return None

        collected: list = []
        for inst in instances.values():
            if not isinstance(inst, dict):
                continue
            individual = inst.get("individual")
            if individual is None:
                continue

            if key == "lira":
                raw = individual.get("score")
                if raw is not None:
                    try:
                        collected.append([max(0.0, min(1.0, float(s))) for s in raw])
                    except (TypeError, ValueError) as exc:
                        logger.warning(
                            "Non-numeric lira score in report (%s); skipping.", exc
                        )
            elif key == "qmia":
                raw = individual.get("member_prob")
                if raw is not None:
                    try:
                        collected.append([max(0.0, min(1.0, float(s))) for s in raw])
                    except (TypeError, ValueError) as exc:
                        logger.warning(
                            "Non-numeric qmia score in report (%s); skipping.", exc
                        )
            elif key == "structural":
                k = individual.get("k_anonymity")
                cd = individual.get("class_disclosure")
                sg = individual.get("smallgroup_risk")
                if k is not None and cd is not None and sg is not None:
                    collected.append(
                        {
                            "k_anonymity": k,
                            "class_disclosure": cd,
                            "smallgroup_risk": sg,
                        }
                    )

        if not collected:
            logger.warning(
                "No individual scores found for %s in report; "
                "ensure the attack was run with report_individual=True.",
                key,
            )
            return None

        return collected

    # ------------------------------------------------------------------
    # Sub-attack execution
    # ------------------------------------------------------------------

    def _run_new_attacks(
        self,
        target: Target,
        existing_mia: dict[str, list],
        existing_struct: dict[str, list],
        mia_scores: dict[str, list],
        structural_scores: dict[str, list],
    ) -> None:
        """Execute sub-attacks that are not already present and populate score dicts.

        When ``behaviour`` is ``'fill_missing'``, attacks found in *existing_mia*
        or *existing_struct* are skipped.  Structural attacks with ``n_reps > 1``
        are clamped to a single run (a warning is logged) because they are
        deterministic.
        """
        for name, params, n_reps in self.attacks:
            if self.behaviour == BEHAVIOUR_FILL_MISSING and (
                name in existing_mia or name in existing_struct
            ):
                logger.info(
                    "Skipping %s - already present in %r.", name, self.report_dir
                )
                continue

            effective_n_reps = n_reps
            if name == "structural" and n_reps > 1:
                logger.warning(
                    "Structural attack is deterministic; n_reps=%d requested "
                    "but all repetitions will be identical. Running once only.",
                    n_reps,
                )
                effective_n_reps = 1

            for rep in range(effective_n_reps):
                logger.info("Running %s (rep %d/%d)", name, rep + 1, effective_n_reps)
                attack_obj = self._run_sub_attack(name, params, target, rep)
                if attack_obj is None:
                    continue

                if name in MIA_ATTACKS:
                    scores = self._extract_mia_scores(attack_obj, name)
                    if scores is not None:
                        mia_scores.setdefault(name, []).append(scores)
                else:
                    scores_struct = self._extract_structural_scores(attack_obj)
                    if scores_struct is not None:
                        structural_scores.setdefault(name, []).append(scores_struct)

    def _run_sub_attack(
        self,
        name: str,
        params: dict,
        target: Target,
        run_idx: int,
    ) -> Attack | None:
        """Create, execute, and return a single sub-attack instance.

        Returns ``None`` and logs a warning if the sub-attack produces no
        results, rather than raising an exception.
        """
        from sacroml.attacks.factory import create_attack  # noqa: PLC0415

        sub_params = copy.deepcopy(params)

        sub_params["report_individual"] = True

        sub_dir = os.path.join(self.output_dir, f"{name}_run{run_idx}")
        sub_params["output_dir"] = sub_dir
        sub_params["write_report"] = False

        try:
            attack_obj = create_attack(name, **sub_params)
            result = attack_obj.attack(target)
        except (RuntimeError, ValueError, OSError, TypeError, AssertionError) as exc:
            logger.error(
                "Sub-attack '%s' (run %d) failed with %s: %s",
                name,
                run_idx,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return None
        if not result:
            logger.warning(
                "Sub-attack '%s' (run %d) produced no results; skipping.",
                name,
                run_idx,
            )
            return None
        return attack_obj

    # ------------------------------------------------------------------
    # Score extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_mia_scores(attack_obj: Attack, name: str) -> list[float] | None:
        """Return per-record membership scores from a completed MIA attack.

        Returns ``None`` and logs a warning when individual scores are absent,
        rather than raising an exception.
        """
        field = _MIA_SCORE_FIELDS[name]

        for metrics_dict in attack_obj.attack_metrics:
            scores = metrics_dict.get("individual", {}).get(field)
            if scores is None:
                continue
            try:
                return [max(0.0, min(1.0, float(s))) for s in scores]
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "%s attack has non-numeric individual scores (%s); skipping.",
                    name,
                    exc,
                )
                return None

        logger.warning(
            "%s attack did not produce individual scores. "
            "Ensure report_individual=True was set.",
            name,
        )
        return None

    @staticmethod
    def _extract_structural_scores(attack_obj: Attack) -> dict | None:
        """Return per-record structural risk indicators, or ``None`` on failure.

        Reads directly from the ``record_level_results`` dataclass, which is
        populated after a successful attack run regardless of ``report_individual``.
        Returns ``None`` and logs a warning when results are unavailable.
        """
        rlr = getattr(attack_obj, "record_level_results", None)
        if rlr is None:
            logger.warning("Structural attack has no record_level_results; skipping.")
            return None
        return {
            "k_anonymity": rlr.k_anonymity,
            "class_disclosure": rlr.class_disclosure,
            "smallgroup_risk": rlr.smallgroup_risk,
        }

    # ------------------------------------------------------------------
    # DataFrame construction
    # ------------------------------------------------------------------

    def _build_dataframe(
        self,
        n_train: int,
        n_test: int,
        mia_scores: dict[str, list[list[float]]],
        structural_scores: dict[str, list[dict]],
    ) -> pd.DataFrame:
        """Assemble the per-record vulnerability DataFrame."""
        n_total = n_train + n_test
        data: dict[str, list] = {}

        data["is_member"] = [1] * n_train + [0] * n_test

        # --- Level 1: within-attack aggregation ---

        mia_mean_cols: list[str] = []

        for name, reps in mia_scores.items():
            scores_array = np.array(reps)  # shape: (n_reps, n_total)

            col_mean = f"{name}_mean"
            col_std = f"{name}_std"
            col_cons = f"{name}_consistency"
            col_vuln = f"{name}_vuln"

            data[col_mean] = np.mean(scores_array, axis=0).tolist()
            data[col_std] = np.std(scores_array, axis=0).tolist()
            data[col_cons] = np.mean(scores_array > self.mia_threshold, axis=0).tolist()
            data[col_vuln] = [m > self.mia_threshold for m in data[col_mean]]

            mia_mean_cols.append(col_mean)

        for _, reps in structural_scores.items():
            if len(reps) == 1:
                k_vals = reps[0]["k_anonymity"]
                cd_vals = reps[0]["class_disclosure"]
                sg_vals = reps[0]["smallgroup_risk"]
            else:
                k_stack = np.array([r["k_anonymity"] for r in reps])
                cd_stack = np.array([r["class_disclosure"] for r in reps])
                sg_stack = np.array([r["smallgroup_risk"] for r in reps])

                k_vals = np.round(np.mean(k_stack, axis=0)).astype(int).tolist()
                cd_vals = (np.mean(cd_stack, axis=0) > 0.5).tolist()
                sg_vals = (np.mean(sg_stack, axis=0) > 0.5).tolist()

            nan_pad = [float("nan")] * n_test
            none_pad = [None] * n_test

            data["struct_k"] = list(k_vals) + nan_pad
            data["struct_cd"] = list(cd_vals) + none_pad
            data["struct_sg"] = list(sg_vals) + none_pad
            data["struct_vuln"] = [
                (k < self.k_threshold or cd or sg)
                for k, cd, sg in zip(k_vals, cd_vals, sg_vals, strict=True)
            ] + none_pad

        # --- Level 2: cross-attack aggregation ---

        if mia_mean_cols:
            mia_means = np.column_stack([data[col] for col in mia_mean_cols])

            data["mia_mean"] = np.mean(mia_means, axis=1).tolist()
            data["mia_gmean"] = np.exp(
                np.mean(np.log(mia_means + _EPS), axis=1)
            ).tolist()

        vuln_cols = [c for c in data if c.endswith("_vuln")]
        n_vuln = np.zeros(n_total)
        for col in vuln_cols:
            vals = data[col]
            for i, v in enumerate(vals):
                if v:
                    n_vuln[i] += 1
        data["n_vulnerable"] = n_vuln.astype(int).tolist()

        df = pd.DataFrame(data)
        df.index = pd.Index([f"record_{i}" for i in range(n_total)], name="record")

        logger.info(
            "Vulnerability matrix: %d records, %d columns", len(df), len(df.columns)
        )
        return df

    # ------------------------------------------------------------------
    # Global metrics and reporting
    # ------------------------------------------------------------------

    def _compute_global_metrics(self, n_train: int, n_test: int) -> None:
        """Compute meta-attack global metrics from the vulnerability DataFrame."""
        if self.vulnerability_df is None:
            raise RuntimeError(
                "_compute_global_metrics called before vulnerability_df was built."
            )
        df = self.vulnerability_df
        membership = np.array([1] * n_train + [0] * n_test)

        if "mia_mean" in df.columns:
            mia_means = df["mia_mean"].to_numpy()
            y_pred_proba = np.column_stack([1 - mia_means, mia_means])
            self.attack_metrics = [metrics.get_metrics(y_pred_proba, membership)]
        else:
            n_vuln_train = int(df.loc[df["is_member"] == 1, "n_vulnerable"].sum())
            self.attack_metrics = [
                {
                    "n_train": n_train,
                    "n_test": n_test,
                    "n_vulnerable_train": n_vuln_train,
                }
            ]

    def _construct_metadata(self) -> None:
        """Add meta-attack specific fields to the report metadata."""
        if self.vulnerability_df is None:
            raise RuntimeError(
                "_construct_metadata called before vulnerability_df was built."
            )
        super()._construct_metadata()
        m = self.attack_metrics[0]
        gm = self.metadata["global_metrics"]

        gm["mia_threshold"] = self.mia_threshold
        gm["k_threshold"] = self.k_threshold
        gm["n_records"] = len(self.vulnerability_df)

        if "AUC" in m:
            gm["AUC"] = m["AUC"]
            gm["TPR"] = m["TPR"]
            gm["Advantage"] = m["Advantage"]

        df = self.vulnerability_df
        n_vuln_cols = len([c for c in df.columns if c.endswith("_vuln")])
        n_all = int((df["n_vulnerable"] == n_vuln_cols).sum()) if n_vuln_cols > 0 else 0
        gm["n_vulnerable_all_attacks"] = n_all

    def _get_attack_metrics_instances(self) -> dict:
        """Return metrics structured for the JSON report."""
        if self.vulnerability_df is None:
            raise RuntimeError(
                "_get_attack_metrics_instances called before"
                " vulnerability_df was built."
            )
        instance = dict(self.attack_metrics[0])

        instance["sub_attacks"] = {
            name: {"n_reps": n_reps} for name, _, n_reps in self.attacks
        }
        instance["individual"] = self.vulnerability_df.to_dict(orient="list")

        return {
            "attack_instance_logger": {"instance_0": instance},
        }

    def _write_report(self, output: dict) -> None:
        """Write JSON report, PDF, and vulnerability matrix CSV.

        By default, append the MetaAttack section to
        ``{report_dir}/report.json`` so it joins any sub-attack reports
        already there.  With ``keep_separate=True``, fall back to the base
        class behaviour and write a standalone ``{output_dir}/report.json``.
        The CSV always lands in ``{output_dir}/vulnerability_matrix.csv``.
        """
        if self.write_report:
            if self.keep_separate:
                super()._write_report(output)
            else:
                self._write_to_report_dir(output)

        if self.write_report and self.vulnerability_df is not None:
            csv_path = os.path.join(self.output_dir, "vulnerability_matrix.csv")
            try:
                self.vulnerability_df.to_csv(csv_path)
                logger.info("Saved vulnerability matrix to %s", csv_path)
            except OSError as exc:
                logger.error(
                    "Failed to write vulnerability matrix to %s: %s",
                    csv_path,
                    exc,
                    exc_info=True,
                )

    def _write_to_report_dir(self, output: dict) -> None:
        """Append MetaAttack JSON (and write PDF) to ``{report_dir}``.

        Uses ``report.write_json`` which appends to an existing
        ``report.json`` if present (via ``GenerateJSONModule``).
        """
        from sacroml.attacks import report  # noqa: PLC0415

        os.makedirs(self.report_dir, exist_ok=True)
        dest: str = os.path.join(self.report_dir, "report")
        logger.info("Appending report: %s.json", dest)
        report.write_json(output, dest)
        pdf_report = self._make_pdf(output)
        if pdf_report is not None:
            report.write_pdf(dest, pdf_report)

    def _make_pdf(self, output: dict) -> FPDF | None:
        """Build the MetaAttack PDF report.

        Delegates to :func:`sacroml.attacks.report.create_meta_report` for
        consistency with the other attacks (see ``create_lr_report``,
        ``create_mia_report``).  The report contains title, attack
        parameters, global metrics, a per-sub-attack summary, and a bar
        chart of records grouped by the number of attacks flagging them.
        """
        from sacroml.attacks import report  # noqa: PLC0415

        return report.create_meta_report(output)

    def __str__(self) -> str:
        """Return a human-readable name for this attack."""
        return "Meta Attack"
