"""Privacy protected TensorFlow model."""

from tensorflow_privacy import DPModel

from aisdc.safemodel.safemodel import SafeModel


class SafeTFModel(SafeModel, DPModel):
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
