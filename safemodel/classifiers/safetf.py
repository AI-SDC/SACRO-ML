import tensorflow as tf
import tensorflow_privacy as tf_privacy

from tensorflow.keras import Model as KerasModel
from safemodel.safemodel import SafeModel  
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras

import tensorflow_privacy as tf_privacy
from tensorflow_privacy import DPModel
from typing import Any


class Safe_tf_DPModel(SafeModel, DPModel):
    """ Privacy Protected tensorflow_privacy DP-SGD subclass of Keras model"""

    def __init__(l2_norm_clip:float, noise_multiplier:float, use_xla:bool=True, *args:any, **kwargs:any) ->None:
        """creates model and applies constraints to parameters"""
        safemodel.__init__(self)
        DPModel.__init__(self, **kwargs)
        self.model_type: str = "tf_DPModel"
        super().preliminary_check(apply_constraints=True, verbose=True)
