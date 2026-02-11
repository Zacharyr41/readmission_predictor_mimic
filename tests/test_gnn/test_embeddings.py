"""Tests for SapBERT concept embeddings."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.gnn.embeddings import SapBERTEmbedder


# ---- fixtures ----


@pytest.fixture
def fake_model():
    """Patch transformers to return a deterministic fake model and tokenizer."""
    torch.manual_seed(42)

    def _fake_tokenizer(texts, *, padding=True, truncation=True, max_length=64, return_tensors="pt"):
        batch_size = len(texts)
        seq_len = 8
        return {
            "input_ids": torch.ones(batch_size, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        }

    def _fake_forward(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        seq_len = kwargs["input_ids"].shape[1]
        torch.manual_seed(42)
        hidden = torch.randn(batch_size, seq_len, 768)
        return SimpleNamespace(last_hidden_state=hidden)

    mock_tokenizer_cls = MagicMock()
    mock_tokenizer_instance = MagicMock(side_effect=_fake_tokenizer)
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance

    mock_model_cls = MagicMock()
    mock_model_instance = MagicMock(side_effect=_fake_forward)
    mock_model_instance.eval.return_value = None
    mock_model_cls.from_pretrained.return_value = mock_model_instance

    with (
        patch("src.gnn.embeddings.AutoTokenizer", mock_tokenizer_cls),
        patch("src.gnn.embeddings.AutoModel", mock_model_cls),
    ):
        yield {
            "tokenizer_cls": mock_tokenizer_cls,
            "model_cls": mock_model_cls,
            "tokenizer": mock_tokenizer_instance,
            "model": mock_model_instance,
        }


# ---- Test 1: Embedding dimensions (real model) ----


@pytest.mark.slow
def test_embedding_dimensions():
    """Real SapBERT produces (N, 768) with no NaN values."""
    embedder = SapBERTEmbedder()
    result = embedder.embed_terms(["Aspirin", "Creatinine", "Stroke"])

    assert result.shape == (3, 768)
    assert not torch.isnan(result).any()


# ---- Test 2: Semantic similarity (real model) ----


@pytest.mark.slow
def test_semantic_similarity():
    """Similar clinical terms cluster; unrelated terms are distant."""
    embedder = SapBERTEmbedder()
    result = embedder.embed_terms(["Aspirin", "Acetylsalicylic acid", "Automobile"])

    cos = torch.nn.functional.cosine_similarity
    sim_related = cos(result[0].unsqueeze(0), result[1].unsqueeze(0)).item()
    sim_unrelated = cos(result[0].unsqueeze(0), result[2].unsqueeze(0)).item()

    assert sim_related > sim_unrelated, (
        f"Synonyms ({sim_related:.3f}) should be more similar than "
        f"unrelated terms ({sim_unrelated:.3f})"
    )
    assert sim_related > 0.5, f"Expected >0.5 for synonyms, got {sim_related}"
    assert sim_unrelated < 0.5, f"Expected <0.5 for unrelated, got {sim_unrelated}"


# ---- Test 3: Caching round-trip (mocked) ----


def test_caching_roundtrip(fake_model, tmp_path):
    """embed_and_cache writes to disk and reloads without touching the model."""
    cache_file = tmp_path / "cache" / "test.pt"
    terms = ["Alpha", "Beta"]
    labels = ["L1", "L2"]

    embedder = SapBERTEmbedder()
    first = embedder.embed_and_cache(terms, labels, cache_file)

    assert cache_file.exists()
    assert set(first.keys()) == {"L1", "L2"}
    assert first["L1"].shape == (768,)

    # Reset mock call counts
    fake_model["tokenizer_cls"].from_pretrained.reset_mock()
    fake_model["model_cls"].from_pretrained.reset_mock()

    # Second call should load from cache, not from model
    embedder2 = SapBERTEmbedder()
    second = embedder2.embed_and_cache(terms, labels, cache_file)

    fake_model["tokenizer_cls"].from_pretrained.assert_not_called()
    fake_model["model_cls"].from_pretrained.assert_not_called()

    for label in labels:
        assert torch.allclose(first[label], second[label])


# ---- Test 4: Batch processing (mocked) ----


def test_batch_processing(fake_model):
    """200 terms are batched correctly and produce (200, 768)."""
    embedder = SapBERTEmbedder()
    terms = [f"term_{i}" for i in range(200)]
    result = embedder.embed_terms(terms)

    assert result.shape == (200, 768)


# ---- Test 5: Empty input ----


def test_empty_input():
    """embed_terms([]) returns shape (0, 768) without loading the model."""
    embedder = SapBERTEmbedder()
    result = embedder.embed_terms([])

    assert result.shape == (0, 768)
