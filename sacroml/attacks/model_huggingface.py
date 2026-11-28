"""Hugging Face sequence-classification target models."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch.nn.functional import softmax

from sacroml.attacks.model import Model, ModelTask


class HuggingFaceModel(Model):
    """Interface to Hugging Face single-label sequence-classification models.

    Install ``transformers`` to load saved models.
    Accepts raw text or two-dimensional arrays of token IDs; an attention mask
    is inferred from the tokenizer's padding token. Direct inference methods also
    accept a mapping of tokenized inputs, including an explicit attention mask.
    Labels must be integer indices into the model's output probabilities.
    Fine-tuning and attacks requiring shadow-model training are not supported.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        model_path: str = "",
        model_module_path: str = "",
        model_name: str = "",
        model_params: dict | None = None,
        train_module_path: str = "",
        train_params: dict | None = None,
        batch_size: int = 32,
        model_task: str = "",
    ) -> None:
        """Instantiate a Hugging Face model and its tokenizer."""
        if model_task and model_task != ModelTask.CLASSIFIER:
            raise ValueError("Only Hugging Face classifiers are supported.")
        super().__init__(
            model=model,
            model_path=model_path,
            model_module_path=model_module_path,
            model_name=model_name,
            model_task=ModelTask.CLASSIFIER,
            model_params=model_params,
            train_module_path=train_module_path,
            train_params=train_params,
        )
        if not callable(getattr(tokenizer, "save_pretrained", None)):
            raise TypeError("Hugging Face tokenizer must implement save_pretrained().")
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero.")
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def get_generalisation_gap(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> float:
        """Return test error minus train error."""
        train = self.score(X_train, y_train)
        test = self.score(X_test, y_test)
        return (1.0 - test) - (1.0 - train)

    def get_losses(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Return one minus the probability assigned to each true label."""
        label_indices = self.get_label_indices(labels)
        probabilities = self.predict_proba(data)
        return 1.0 - probabilities[np.arange(len(data)), label_indices]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy for a set of samples."""
        return float(np.mean(self.predict(X) == y))

    def predict(self, X: np.ndarray | Mapping[str, np.ndarray]) -> np.ndarray:
        """Return predicted class indices."""
        return np.argmax(self.predict_proba(X), axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> HuggingFaceModel:
        """Raise an explicit error because fine-tuning is task-specific."""
        del X, y
        raise NotImplementedError(
            "Hugging Face model fine-tuning is not implemented. Train the model "
            "with a task-specific routine before using it as a target."
        )

    def clone(self) -> HuggingFaceModel:
        """Return a deep copy of the model and tokenizer."""
        return HuggingFaceModel(
            model=deepcopy(self.model),
            tokenizer=deepcopy(self.tokenizer),
            model_path=self.model_path,
            model_module_path=self.model_module_path,
            model_name=self.model_name,
            model_params=self.model_params,
            train_module_path=self.train_module_path,
            train_params=self.train_params,
            batch_size=self.batch_size,
        )

    def _prepare_inputs(
        self, X: np.ndarray | Mapping[str, np.ndarray]
    ) -> tuple[list[str] | np.ndarray | dict[str, np.ndarray], int]:
        """Validate inputs without changing the order of samples."""
        if isinstance(X, Mapping):
            if "input_ids" not in X:
                raise ValueError("Tokenized inputs must contain input_ids.")
            encoded = {key: np.asarray(value) for key, value in X.items()}
            shape = encoded["input_ids"].shape
            if len(shape) != 2 or any(
                value.shape != shape for value in encoded.values()
            ):
                raise ValueError("Tokenized input arrays must have matching 2D shapes.")
            if any(
                not np.issubdtype(value.dtype, np.integer) for value in encoded.values()
            ):
                raise ValueError("Tokenized input arrays must contain integers.")
            return encoded, shape[0]

        inputs = np.asarray(X)
        if inputs.ndim == 0:
            raise ValueError("Expected a batch of text or token ID sequences.")
        is_text = inputs.dtype.kind in "US" or (
            inputs.dtype.kind == "O"
            and all(isinstance(item, str) for item in inputs.reshape(-1))
        )
        if is_text and (
            inputs.ndim == 1 or (inputs.ndim == 2 and inputs.shape[1] == 1)
        ):
            samples = inputs.reshape(-1).tolist()
            return samples, len(samples)
        if inputs.ndim == 2 and np.issubdtype(inputs.dtype, np.integer):
            return inputs, len(inputs)
        raise ValueError("Expected a 1D text array or 2D integer token IDs.")

    def _prepare_batch(
        self, inputs: list[str] | np.ndarray | dict[str, np.ndarray], start: int
    ) -> dict[str, torch.Tensor]:
        """Convert a batch to the model's keyword arguments."""
        end = start + self.batch_size
        if isinstance(inputs, dict):
            return {
                key: torch.as_tensor(value[start:end], dtype=torch.long)
                for key, value in inputs.items()
            }
        if isinstance(inputs, list):
            return self.tokenizer(
                inputs[start:end], padding=True, truncation=True, return_tensors="pt"
            )
        input_ids = torch.as_tensor(inputs[start:end], dtype=torch.long)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        attention_mask = (
            input_ids.ne(pad_token_id).long()
            if pad_token_id is not None
            else torch.ones_like(input_ids)
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def predict_proba(self, X: np.ndarray | Mapping[str, np.ndarray]) -> np.ndarray:
        """Return class probabilities for raw text or token ID arrays."""
        num_labels = self._num_labels()
        inputs, size = self._prepare_inputs(X)
        if size == 0:
            return np.empty((0, num_labels), dtype=np.float64)

        model_device = next(self.model.parameters(), torch.empty(0)).device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        was_training = self.model.training
        self.model.to(device)
        self.model.eval()
        all_probs: list[np.ndarray] = []
        try:
            with torch.no_grad():
                for start in range(0, size, self.batch_size):
                    batch = self._prepare_batch(inputs, start)
                    model_inputs = {
                        key: value.to(device) for key, value in batch.items()
                    }
                    output = self.model(**model_inputs)
                    logits = getattr(output, "logits", None)
                    if not isinstance(logits, torch.Tensor) or logits.shape != (
                        min(self.batch_size, size - start),
                        num_labels,
                    ):
                        raise ValueError(
                            "Expected sequence-classification logits with shape "
                            "(batch_size, num_labels)."
                        )
                    if not torch.isfinite(logits).all():
                        raise ValueError("Model produced non-finite logits.")
                    all_probs.append(softmax(logits.float(), dim=1).cpu().numpy())
        finally:
            self.model.to(model_device)
            self.model.train(was_training)

        return np.concatenate(all_probs, axis=0).astype(np.float64)

    def get_classes(self) -> np.ndarray:
        """Return class indices configured on the model."""
        return np.arange(self._num_labels())

    def set_params(self, **kwargs: object) -> HuggingFaceModel:
        """Set the model random seed when requested."""
        if "random_state" in kwargs:
            seed = kwargs["random_state"]
            if not isinstance(seed, int):
                raise TypeError("random_state must be an integer.")
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        return self

    def get_params(self) -> dict:
        """Return parameters recorded for model construction."""
        return self.model_params

    def save(self, path: str) -> None:
        """Save the model and tokenizer using Hugging Face serialization."""
        save_pretrained = getattr(self.model, "save_pretrained", None)
        if not callable(save_pretrained):
            raise TypeError("Hugging Face model must implement save_pretrained().")
        save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(
        cls,
        model_path: str,
        model_module_path: str = "",
        model_name: str = "",
        model_params: dict | None = None,
        train_module_path: str = "",
        train_params: dict | None = None,
        batch_size: int = 32,
        model_task: str = "",
    ) -> HuggingFaceModel:
        """Load a sequence-classification model and tokenizer from a directory."""
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise ImportError(
                "Loading Hugging Face models requires the 'transformers' dependency."
            ) from error

        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_path=model_path,
            model_module_path=model_module_path,
            model_name=model_name,
            model_params=model_params,
            train_module_path=train_module_path,
            train_params=train_params,
            batch_size=batch_size,
            model_task=model_task,
        )

    def _num_labels(self) -> int:
        """Get the number of labels from the model configuration."""
        num_labels = getattr(getattr(self.model, "config", None), "num_labels", None)
        problem_type = getattr(
            getattr(self.model, "config", None), "problem_type", None
        )
        if (
            not isinstance(num_labels, int)
            or num_labels < 2
            or problem_type
            in (
                "regression",
                "multi_label_classification",
            )
        ):
            raise ValueError(
                "Expected a single-label classification model with at least two labels."
            )
        return num_labels
