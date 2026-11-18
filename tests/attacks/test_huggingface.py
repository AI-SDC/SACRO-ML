"""Tests for Hugging Face model support."""

from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from sacroml.attacks.likelihood_attack import LIRAAttack
from sacroml.attacks.model import ModelTask
from sacroml.attacks.model_huggingface import HuggingFaceModel
from sacroml.attacks.model_pytorch import PytorchModel
from sacroml.attacks.target import Target
from sacroml.attacks.worst_case_attack import WorstCaseAttack


@pytest.fixture(autouse=True)
def preserve_random_state():
    """Keep these tests from changing random seeds used by other tests."""
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        yield
    np.random.set_state(numpy_state)


class FakeTokenizer:
    """Minimal tokenizer implementation for wrapper tests."""

    pad_token_id = 0

    def __call__(self, texts, **_kwargs):
        """Tokenize each string into a fixed-width sequence."""
        input_ids = torch.tensor([[len(text), 1] for text in texts])
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def save_pretrained(self, path):
        """Write a marker file to the model directory."""
        with open(f"{path}/tokenizer.txt", "w", encoding="utf-8") as file:
            file.write("tokenizer")


class FakeHFModel(torch.nn.Module):
    """Minimal sequence-classification model for wrapper tests."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_labels=2)

    def forward(self, input_ids, attention_mask):
        """Return deterministic logits derived from token IDs."""
        values = (input_ids * attention_mask).sum(dim=1).float()
        return SimpleNamespace(logits=torch.stack((values, -values), dim=1))

    def save_pretrained(self, path):
        """Write a marker file to the model directory."""
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/model.txt", "w", encoding="utf-8") as file:
            file.write("model")


def test_target_accepts_huggingface_text_and_token_inputs():
    """Target wraps a HF model and supports both input representations."""
    target = Target(model=FakeHFModel(), tokenizer=FakeTokenizer(), batch_size=1)

    assert isinstance(target.model, HuggingFaceModel)
    assert target.model.model_task == ModelTask.CLASSIFIER
    text = np.array(["hello", "a"])
    token_ids = np.array([[5, 1], [1, 1]], dtype=np.int64)
    assert np.allclose(
        target.model.predict_proba(text), target.model.predict_proba(token_ids)
    )
    assert np.array_equal(target.model.predict(text), [0, 0])


def test_target_huggingface_save_load_round_trip(tmp_path, monkeypatch):
    """Target save/load persists the HF model and its tokenizer together."""

    class FakeAutoModel:
        """Test substitute for the Transformers model loader."""

        @staticmethod
        def from_pretrained(_path, local_files_only):
            """Return a fake model."""
            assert local_files_only
            return FakeHFModel()

    class FakeAutoTokenizer:
        """Test substitute for the Transformers tokenizer loader."""

        @staticmethod
        def from_pretrained(_path, local_files_only):
            """Return a fake tokenizer."""
            assert local_files_only
            return FakeTokenizer()

    transformers = ModuleType("transformers")
    transformers.AutoModelForSequenceClassification = FakeAutoModel
    transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    target = Target(model=FakeHFModel(), tokenizer=FakeTokenizer())
    target_path = tmp_path / "target"
    target.save(str(target_path))
    assert (target_path / "model" / "model.txt").is_file()
    assert (target_path / "model" / "tokenizer.txt").is_file()

    loaded = Target()
    loaded.load(str(target_path))
    assert isinstance(loaded.model, HuggingFaceModel)
    assert loaded.model.model_task == ModelTask.CLASSIFIER
    assert loaded.tokenizer is loaded.model.tokenizer
    assert np.allclose(
        loaded.model.predict_proba(np.array(["hello"])),
        target.model.predict_proba(np.array(["hello"])),
    )


def test_target_requires_tokenizer_for_huggingface_model():
    """Target requires the tokenizer needed for model inference."""
    transformers = pytest.importorskip("transformers")
    model = transformers.BertForSequenceClassification(
        transformers.BertConfig(
            vocab_size=4,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
    )
    with pytest.raises(ValueError, match="tokenizer is required"):
        Target(model=model)


def test_pytorch_model_with_config_and_save_method_is_still_pytorch():
    """Unrelated PyTorch modules with similar APIs keep their existing wrapper."""
    assert isinstance(Target(model=FakeHFModel()).model, PytorchModel)


def test_huggingface_rejects_non_classification_task():
    """The classifier wrapper must not silently accept regression targets."""
    with pytest.raises(ValueError, match="Only Hugging Face classifiers"):
        Target(model=FakeHFModel(), tokenizer=FakeTokenizer(), model_task="regressor")


def test_tokenized_inputs_use_explicit_attention_mask():
    """Pass all tokenized fields to the model without inferring the mask."""
    wrapper = HuggingFaceModel(FakeHFModel(), FakeTokenizer())
    encoded = {
        "input_ids": np.array([[5, 0], [1, 1]]),
        "attention_mask": np.array([[0, 1], [1, 1]]),
    }
    probabilities = wrapper.predict_proba(encoded)
    assert probabilities[0, 0] == pytest.approx(0.5)
    assert probabilities[1, 0] > probabilities[0, 0]


def test_invalid_inputs_and_unsupported_classification():
    """Reject malformed token arrays and unsupported classification tasks."""
    model = FakeHFModel()
    wrapper = HuggingFaceModel(model, FakeTokenizer())
    with pytest.raises(ValueError, match="integer"):
        wrapper.predict_proba(np.array([[1.5, 2.0]]))
    with pytest.raises(ValueError, match="matching 2D shapes"):
        wrapper.predict_proba(
            {"input_ids": np.ones((2, 3)), "attention_mask": np.ones((2, 2))}
        )
    model.config.problem_type = "multi_label_classification"
    with pytest.raises(ValueError, match="single-label classification"):
        wrapper.predict_proba(np.array(["hello"]))


def test_real_transformers_model_round_trip(tmp_path):
    """Check local HF model and tokenizer inference before and after saving."""
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    unknown, padding = "[UNK]", "[PAD]"
    tokenizer_impl = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {unknown: 0, padding: 1, "hello": 2, "world": 3},
            unk_token=unknown,
        )
    )
    tokenizer_impl.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl, unk_token=unknown, pad_token=padding
    )
    config = transformers.BertConfig(
        vocab_size=4,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        num_labels=2,
    )
    model = transformers.BertForSequenceClassification(config)
    text = np.array(["hello world", "hello"])
    labels = np.array([0, 1])
    target = Target(
        model=model,
        tokenizer=tokenizer,
        batch_size=1,
        X_train=text,
        y_train=labels,
        X_test=text,
        y_test=labels,
    )
    assert isinstance(target.model, HuggingFaceModel)
    probabilities = target.model.predict_proba(text)
    inputs = tokenizer(list(text), padding=True, return_tensors="np")
    assert np.allclose(
        target.model.predict_proba(dict(inputs)), probabilities, atol=1e-6
    )
    assert np.allclose(
        target.model.predict_proba(inputs["input_ids"]), probabilities, atol=1e-6
    )
    assert model.training
    target.save(str(tmp_path / "target"))
    loaded = Target()
    loaded.load(str(tmp_path / "target"))
    assert loaded.batch_size == 1
    assert np.array_equal(loaded.X_train, text)
    assert np.allclose(loaded.model.predict_proba(text), probabilities, atol=1e-6)


def test_lira_rejects_huggingface_target(caplog):
    """LiRA cannot retrain Hugging Face shadow models."""
    target = Target(
        model=FakeHFModel(),
        tokenizer=FakeTokenizer(),
        X_train=np.array(["hello"]),
        y_train=np.array([0]),
        X_test=np.array(["world"]),
        y_test=np.array([0]),
    )
    assert not LIRAAttack.attackable(target)
    assert "shadow-model training" in caplog.text


def test_worst_case_attack_with_huggingface_target(tmp_path):
    """Run an attack against the Hugging Face probability interface."""
    text = np.array([f"record {i}" for i in range(40)])
    target = Target(
        model=FakeHFModel(),
        tokenizer=FakeTokenizer(),
        X_train=text[:20],
        y_train=np.zeros(20, dtype=int),
        X_test=text[20:],
        y_test=np.zeros(20, dtype=int),
    )
    attack = WorstCaseAttack(
        output_dir=str(tmp_path / "output"),
        write_report=False,
        n_reps=1,
        n_dummy_reps=1,
        test_prop=0.5,
    )
    report = attack.attack(target)
    assert report["metadata"]["attack_name"] == "WorstCase attack"
