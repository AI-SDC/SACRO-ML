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

import copy
import logging
import os

import numpy as np
import pandas as pd
from fpdf import FPDF

from sacroml import metrics
from sacroml.attacks.attack import Attack
from sacroml.attacks.target import Target

logging.basicConfig(level=logging.INFO)
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
        """Run all sub-attacks and aggregate per-record vulnerabilities.

        For each attack specification the method:
        1. Runs the sub-attack *n_reps* times, each in an isolated subdirectory.
        2. Extracts per-record scores from each run.
        """
        # {name: [[scores_rep0], [scores_rep1], ...]}  for MIA
        # {name: [{"k_anonymity": [...], ...}, ...]}   for structural
        mia_scores: dict[str, list[list[float]]] = {}
        structural_scores: dict[str, list[dict]] = {}

        for name, params, n_reps in self.attacks:
            for rep in range(n_reps):
                logger.info(
                    "Running %s (rep %d/%d)", name, rep + 1, n_reps
                )
                attack_obj = self._run_sub_attack(name, params, target, rep)

                if name in self.MIA_ATTACKS:
                    scores = self._extract_mia_scores(attack_obj, name)
                    mia_scores.setdefault(name, []).append(scores)
                else:
                    scores = self._extract_structural_scores(attack_obj)
                    structural_scores.setdefault(name, []).append(scores)

        n_train = len(target.X_train)
        n_test = len(target.X_test)
        self.vulnerability_df = self._build_dataframe(
            n_train, n_test, mia_scores, structural_scores
        )

        # Compute global metrics using the aggregated MIA mean as a
        # membership predictor.  If no MIA attacks were run (structural
        # only), store a summary dict without standard MIA metrics.
        self._compute_global_metrics(n_train, n_test)

        output = self._make_report(target)
        self._write_report(output)

        # Save the vulnerability matrix as CSV alongside the JSON report.
        if self.write_report:
            csv_path = os.path.join(self.output_dir, "vulnerability_matrix.csv")
            self.vulnerability_df.to_csv(csv_path)
            logger.info("Saved vulnerability matrix to %s", csv_path)

        return output

    # ------------------------------------------------------------------
    # Sub-attack execution
    # ------------------------------------------------------------------

    def _run_sub_attack(
        self,
        name: str,
        params: dict,
        target: Target,
        run_idx: int,
    ) -> Attack:
        """Create, execute, and return a single sub-attack instance.

        Parameters
        ----------
        name : str
            Attack name as registered in the factory (e.g. ``"lira"``).
        params : dict
            Constructor keyword arguments for the sub-attack.
        target : Target
            The shared target all sub-attacks are evaluated against.
        run_idx : int
            Repetition index, used to create an isolated output subdirectory.

        Returns
        -------
        Attack
            The sub-attack instance after ``.attack(target)`` has been called.
            Per-record scores are accessible on the returned object.
        """
        from sacroml.attacks.factory import create_attack

        sub_params = copy.deepcopy(params)

        # Force per-record reporting on MIA attacks.
        # Structural always computes record_level_results regardless.
        if name in MetaAttack.MIA_ATTACKS:
            sub_params["report_individual"] = True

        # Isolate each run in its own subdirectory under self.output_dir.
        sub_dir = os.path.join(self.output_dir, f"{name}_run{run_idx}")
        sub_params["output_dir"] = sub_dir
        sub_params["write_report"] = False

        attack_obj = create_attack(name, **sub_params)
        result = attack_obj.attack(target)
        if not result:
            raise RuntimeError(
                f"Sub-attack '{name}' (run {run_idx}) produced no results. "
                f"The target may not be attackable by this attack type."
            )
        return attack_obj

    # ------------------------------------------------------------------
    # Score extraction
    # ------------------------------------------------------------------

    _MIA_SCORE_FIELDS: dict[str, str] = {
        "lira": "score",
        "qmia": "member_prob",
    }
    """Maps attack name → key inside the ``"individual"`` dict that holds
    the per-record membership score.

    For LiRA (default ``offline`` mode) the ``"score"`` field stores
    ``norm.cdf(logit, out_mean, out_std)`` — the CDF of the record's
    logit under the non-member distribution.  High values mean the logit
    is unusually high for a non-member, i.e. evidence **for** membership.
    The ``_DummyClassifier.predict`` convention confirms: member when
    ``score > 0.5``.  Non-default Carlini modes may produce scores outside
    [0, 1]; these are clipped during extraction.
    """

    @staticmethod
    def _extract_mia_scores(attack_obj: Attack, name: str) -> list[float]:
        """Return per-record membership scores from a completed MIA attack.

        Parameters
        ----------
        attack_obj : Attack
            A LiRA or QMIA attack instance after ``.attack()`` has run
            with ``report_individual=True``.
        name : str
            Attack name (``"lira"`` or ``"qmia"``), used to look up the
            correct score field.

        Returns
        -------
        list[float]
            One score per record (train then test), values in [0, 1].
        """
        field = MetaAttack._MIA_SCORE_FIELDS[name]

        # LiRA stores metrics as a list; QMIA also uses a list.
        # Both place the "individual" dict in attack_metrics[N].
        for metrics_dict in attack_obj.attack_metrics:
            if "individual" in metrics_dict:
                scores = metrics_dict["individual"][field]
                # Clip to [0, 1]: default offline mode is already bounded,
                # but Carlini modes can produce unbounded log-likelihood ratios.
                return [max(0.0, min(1.0, s)) for s in scores]

        raise RuntimeError(
            f"{name} attack did not produce individual scores. "
            f"Ensure report_individual=True was set."
        )

    @staticmethod
    def _extract_structural_scores(attack_obj: Attack) -> dict:
        """Return per-record structural risk indicators.

        Reads directly from the ``record_level_results`` dataclass, which
        is always populated regardless of ``report_individual``.

        Returns
        -------
        dict
            Keys: ``"k_anonymity"`` (list[int]),
            ``"class_disclosure"`` (list[bool]),
            ``"smallgroup_risk"`` (list[bool]).
            Length = number of training records.
        """
        rlr = attack_obj.record_level_results
        return {
            "k_anonymity": rlr.k_anonymity,
            "class_disclosure": rlr.class_disclosure,
            "smallgroup_risk": rlr.smallgroup_risk,
        }

    # ------------------------------------------------------------------
    # DataFrame construction
    # ------------------------------------------------------------------

    _EPS: float = 1e-10
    """Small constant to avoid log(0) in geometric mean computation."""

    def _build_dataframe(
        self,
        n_train: int,
        n_test: int,
        mia_scores: dict[str, list[list[float]]],
        structural_scores: dict[str, list[dict]],
    ) -> pd.DataFrame:
        """Assemble the per-record vulnerability DataFrame.

        Parameters
        ----------
        n_train, n_test : int
            Number of training / test records in the Target.
        mia_scores : dict
            ``{name: [scores_rep0, scores_rep1, ...]}`` where each
            ``scores_repN`` is a list of floats with length
            ``n_train + n_test``.
        structural_scores : dict
            ``{name: [dict_rep0, dict_rep1, ...]}`` where each dict has
            keys ``k_anonymity``, ``class_disclosure``, ``smallgroup_risk``
            with lists of length ``n_train``.

        Returns
        -------
        pd.DataFrame
        """
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
            data[col_cons] = np.mean(
                scores_array > self.mia_threshold, axis=0
            ).tolist()
            data[col_vuln] = [m > self.mia_threshold for m in data[col_mean]]

            mia_mean_cols.append(col_mean)

        for name, reps in structural_scores.items():
            if len(reps) == 1:
                k_vals = reps[0]["k_anonymity"]
                cd_vals = reps[0]["class_disclosure"]
                sg_vals = reps[0]["smallgroup_risk"]
            else:
                # Average k-anonymity across reps; majority vote for booleans.
                k_stack = np.array([r["k_anonymity"] for r in reps])
                cd_stack = np.array([r["class_disclosure"] for r in reps])
                sg_stack = np.array([r["smallgroup_risk"] for r in reps])

                k_vals = np.round(np.mean(k_stack, axis=0)).astype(int).tolist()
                cd_vals = (np.mean(cd_stack, axis=0) > 0.5).tolist()
                sg_vals = (np.mean(sg_stack, axis=0) > 0.5).tolist()

            # Pad with NaN/None for test records (structural is train-only).
            nan_pad = [float("nan")] * n_test
            none_pad = [None] * n_test

            data["struct_k"] = list(k_vals) + nan_pad
            data["struct_cd"] = list(cd_vals) + none_pad
            data["struct_sg"] = list(sg_vals) + none_pad
            data["struct_vuln"] = [
                (k < self.k_threshold or cd or sg)
                for k, cd, sg in zip(k_vals, cd_vals, sg_vals)
            ] + none_pad

        # --- Level 2: cross-attack aggregation ---

        if mia_mean_cols:
            mia_means = np.column_stack(
                [data[col] for col in mia_mean_cols]
            )  # shape: (n_total, n_mia_attacks)

            data["mia_mean"] = np.mean(mia_means, axis=1).tolist()
            data["mia_gmean"] = np.exp(
                np.mean(np.log(mia_means + self._EPS), axis=1)
            ).tolist()

        # n_vulnerable: count of attacks flagging each record.
        # Use truthiness (not identity) so numpy bools are handled correctly.
        vuln_cols = [c for c in data if c.endswith("_vuln")]
        n_vuln = np.zeros(n_total)
        for col in vuln_cols:
            vals = data[col]
            for i, v in enumerate(vals):
                if v:
                    n_vuln[i] += 1
        data["n_vulnerable"] = n_vuln.astype(int).tolist()

        df = pd.DataFrame(data)
        df.index = [f"record_{i}" for i in range(n_total)]
        df.index.name = "record"

        logger.info(
            "Vulnerability matrix: %d records, %d columns", len(df), len(df.columns)
        )
        return df

    # ------------------------------------------------------------------
    # Global metrics and reporting
    # ------------------------------------------------------------------

    def _compute_global_metrics(self, n_train: int, n_test: int) -> None:
        """Compute meta-attack global metrics from the vulnerability DataFrame.

        When MIA attacks are present, uses ``mia_mean`` as a membership
        predictor and calls :func:`~sacroml.metrics.get_metrics` to obtain
        AUC, TPR, Advantage, etc.  When only structural attacks were run,
        stores a summary dict without standard MIA metrics.
        """
        df = self.vulnerability_df
        membership = np.array([1] * n_train + [0] * n_test)

        if "mia_mean" in df.columns:
            mia_means = df["mia_mean"].values
            y_pred_proba = np.column_stack([1 - mia_means, mia_means])
            self.attack_metrics = [
                metrics.get_metrics(y_pred_proba, membership)
            ]
        else:
            # Structural only — no membership probability to evaluate.
            n_vuln_train = int(
                df.loc[df["is_member"] == 1, "n_vulnerable"].sum()
            )
            self.attack_metrics = [
                {
                    "n_train": n_train,
                    "n_test": n_test,
                    "n_vulnerable_train": n_vuln_train,
                }
            ]

    def _construct_metadata(self) -> None:
        """Add meta-attack specific fields to the report metadata."""
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
        n_all = int((df["n_vulnerable"] == df["n_vulnerable"].max()).sum())
        gm["n_vulnerable_all_attacks"] = n_all

    def _get_attack_metrics_instances(self) -> dict:
        """Return metrics structured for the JSON report.

        Includes the standard metrics dict, a ``sub_attacks`` summary,
        and the full vulnerability DataFrame under ``individual``.
        """
        instance = dict(self.attack_metrics[0])

        # Sub-attack summary: name → {n_reps, ...}
        instance["sub_attacks"] = {
            name: {"n_reps": n_reps}
            for name, _, n_reps in self.attacks
        }

        # Serialise the vulnerability DataFrame as dict-of-lists.
        instance["individual"] = self.vulnerability_df.to_dict(orient="list")

        return {
            "attack_instance_logger": {"instance_0": instance},
        }

    def _make_pdf(self, output: dict) -> FPDF | None:
        """Return ``None`` — PDF generation is not yet implemented."""
        return None

    def __str__(self) -> str:
        return "Meta Attack"
