"""Privacy protected TensorFlow model."""

# pylint: disable=unused-import
from typing import Any

import tensorflow as tf
import tensorflow_privacy as tf_privacy
from tensorflow_privacy import DPModel
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras

from ..safemodel import SafeModel


class Safe_tf_DPModel(SafeModel, DPModel):
    """Privacy Protected tensorflow_privacy DP-SGD subclass of Keras model."""

    # pylint:disable=super-init-not-called
    def __init__(
        self,
        l2_norm_clip: float,
        noise_multiplier: float,
        use_xla: bool,
        *args: any,
        **kwargs: any,
    ) -> None:
        """Create model and apply constraints to parameters."""
        raise NotImplementedError
