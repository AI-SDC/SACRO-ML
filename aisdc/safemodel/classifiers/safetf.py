""" work in progress to allow use of the DPModel classes
 Jim smith 2022
 When ready, linting of  the imports will be enabled
"""


# pylint: disable=unused-import
from typing import Any

import tensorflow as tf
import tensorflow_privacy as tf_privacy
from tensorflow_privacy import DPModel

# from tensorflow.keras import Model as KerasModel
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras

from ..safemodel import SafeModel


class Safe_tf_DPModel(SafeModel, DPModel):
    """Privacy Protected tensorflow_privacy DP-SGD subclass of Keras model"""

    # remove comment once model starts to be populated
    # pylint:disable=super-init-not-called
    def __init__(
        self,
        l2_norm_clip: float,
        noise_multiplier: float,
        use_xla: bool,
        *args: any,
        **kwargs: any
    ) -> None:
        """creates model and applies constraints to parameters"""
        # safemodel.__init__(self)
        # DPModel.__init__(self, **kwargs)
        # self.model_type: str = "tf_DPModel"
        # super().preliminary_check(apply_constraints=True, verbose=True)
        raise NotImplementedError
