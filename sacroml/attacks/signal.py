"""Per-record membership-inference signals.

A *signal* is one scalar per record, computed from a model's behaviour on that
record, where larger values mean "more member-like" (the model fits the record
better). Attacks such as LiRA fit distributions to these scalars across shadow
models, so a good signal is roughly Gaussian across models.

Current signals
---------------
- Classification: ``logit`` -> logit(p_y), the correct-class logit.
- Regression: ``log_mse`` -> -log(per-record MSE).

Adding a signal
---------------
Write a function with signature ``(model, X, y) -> np.ndarray`` and
register it in ``CLASSIFICATION_SIGNALS`` or ``REGRESSION_SIGNALS``. It is then
selectable by name via ``get_signals(..., signal="<name>")``.

Label convention
----------------
For classifiers, ``y`` must be the *real class values* (the values in
``model.get_classes()``), not positional indices. This keeps shadow models
whose training subset lacked some classes safe to score: records whose class
is unknown to the model get the signal of a zero probability.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from sacroml.attacks.model import Model, ModelTask
from sacroml.attacks.utils import EPS, logit

DEFAULT_SIGNAL: str = "default"


def _classification_logit(model: Model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return correct-class logits, using logit(0) for absent shadow classes."""
    class_map = {c: i for i, c in enumerate(model.get_classes())}
    proba = model.predict_proba(X)
    return np.array(
        [
            logit(row[class_map[label]]) if label in class_map else logit(0)
            for row, label in zip(proba, y, strict=True)
        ]
    )


def _regression_log_mse(model: Model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return negative log-MSE signals, with larger values for smaller errors."""
    return -np.log(model.get_losses(X, y) + EPS)


Signal = Callable[[Model, np.ndarray, np.ndarray], np.ndarray]
CLASSIFICATION_SIGNALS: dict[str, Signal] = {"logit": _classification_logit}
REGRESSION_SIGNALS: dict[str, Signal] = {"log_mse": _regression_log_mse}

_REGISTRIES: dict[ModelTask, dict[str, Signal]] = {
    ModelTask.CLASSIFIER: CLASSIFICATION_SIGNALS,
    ModelTask.REGRESSOR: REGRESSION_SIGNALS,
}

_DEFAULTS: dict[ModelTask, str] = {
    ModelTask.CLASSIFIER: "logit",
    ModelTask.REGRESSOR: "log_mse",
}


def get_signals(
    model: Model,
    X: np.ndarray,
    y: np.ndarray,
    signal: str = DEFAULT_SIGNAL,
) -> np.ndarray:
    """Return one signal per record (larger = more member-like).

    Parameters
    ----------
    model : Model
        Target or shadow model wrapper.
    X : np.ndarray
        Features of the records.
    y : np.ndarray
        Labels of the records.
    signal : str
        Signal name, or ``"default"`` for the task default (``"logit"`` for
        classification, ``"log_mse"`` for regression).

    Returns
    -------
    np.ndarray
        Array of shape ``(n_records,)``.
    """
    task = ModelTask(model.model_task)
    if task not in _REGISTRIES:
        raise ValueError(f"Unsupported 'model_task': '{task}'")

    registry = _REGISTRIES[task]
    name = _DEFAULTS[task] if signal == DEFAULT_SIGNAL else signal
    if name not in registry:
        raise ValueError(
            f"Unknown signal '{signal}' for {task.value}. "
            f"Available: {', '.join(sorted(registry))}"
        )
    return np.asarray(registry[name](model, X, y), dtype=float)
