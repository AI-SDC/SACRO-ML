"""Privacy protected TensorFlow model."""

from tensorflow_privacy import DPModel

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
