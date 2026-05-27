"""Shared numerical and default-value constants for the attacks package.

Centralising these here avoids duplication across attack modules and makes
the *why* of each magic number visible at a glance.

Notes
-----
A separate :data:`sacroml.attacks.utils.EPS` (``1e-16``) and an identical
``EPS`` in :mod:`sacroml.attacks.likelihood_attack` are kept independently
for now because they predate this module and migrating them is a wider
refactor.  A follow-up PR can converge those onto a single constant defined
here once the call sites have been audited.
"""

from __future__ import annotations

EPS_META: float = 1e-10
"""Tolerance added before ``log()`` in geometric-mean aggregation.

Looser than :data:`sacroml.attacks.utils.EPS` (``1e-16``) because the
geometric mean of MIA scores in :class:`~sacroml.attacks.meta_attack.MetaAttack`
does not need the same precision as normal-distribution CDF/PDF
calculations and benefits from a value comfortably above floating-point
denormals.
"""

DEFAULT_MIA_THRESHOLD: float = 0.5
"""Default cutoff above which a per-record membership-inference score is
flagged as vulnerable.

Used as the ``mia_threshold`` default for
:class:`~sacroml.attacks.meta_attack.MetaAttack` so the value can be
referenced symbolically from tests, examples, and documentation.
"""
