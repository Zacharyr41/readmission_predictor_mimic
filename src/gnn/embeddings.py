"""SapBERT-based SNOMED concept embeddings.

Generates dense 768-dimensional vector representations for SNOMED-CT concepts
using SapBERT (Self-Alignment Pretraining for Biomedical Entity Representations).
These embeddings serve as initial node features for the GNN prediction pathway.

Reference:
    Liu, F., Shareghi, E., Meng, Z., Basaldella, M., & Collier, N. (2021).
    Self-Alignment Pretraining for Biomedical Entity Representations.
    *NAACL-HLT 2021*, 4228-4238. https://doi.org/10.18653/v1/2021.naacl-main.334

Model: ``cambridgeltl/SapBERT-from-PubMedBERT-fulltext`` on HuggingFace.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph_construction.terminology.snomed_mapper import SnomedMapper

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModel, AutoTokenizer

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


class SapBERTEmbedder:
    """Generate dense 768-dim embeddings for clinical terms using SapBERT.

    SapBERT is a PubMedBERT variant pre-trained on UMLS synonyms, producing
    embeddings where clinically similar terms cluster together.

    The model is loaded lazily on first use and can be pointed at a local
    HuggingFace cache directory.
    """

    MODEL_ID = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    EMBEDDING_DIM = 768

    def __init__(self, cache_dir: Path | None = None) -> None:
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "SapBERTEmbedder requires 'torch' and 'transformers'. "
                "Install with: uv pip install torch transformers"
            )
        self._cache_dir = str(cache_dir) if cache_dir else None
        self._tokenizer = None
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load tokenizer and model on first use."""
        if self._model is not None:
            return
        logger.info("Loading SapBERT model: %s", self.MODEL_ID)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, cache_dir=self._cache_dir
        )
        self._model = AutoModel.from_pretrained(
            self.MODEL_ID, cache_dir=self._cache_dir
        )
        self._model.eval()

    def embed_terms(self, terms: list[str]) -> torch.Tensor:
        """Embed a list of clinical terms into 768-dim vectors.

        Uses CLS pooling over SapBERT's last hidden state.

        Args:
            terms: Clinical term strings to embed.

        Returns:
            Tensor of shape ``(len(terms), 768)``.
        """
        if len(terms) == 0:
            return torch.empty(0, self.EMBEDDING_DIM)

        self._load_model()

        batch_size = 64
        all_embeddings = []

        for i in range(0, len(terms), batch_size):
            batch = [t.lower() for t in terms[i : i + batch_size]]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self._model(**encoded)
            # CLS pooling
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings)

        return torch.cat(all_embeddings, dim=0)

    def embed_and_cache(
        self,
        terms: list[str],
        labels: list[str],
        cache_path: Path,
    ) -> dict[str, torch.Tensor]:
        """Embed terms and cache the result to disk.

        If ``cache_path`` already exists, loads and returns the cached dict
        without touching the model.

        Args:
            terms: Clinical term strings to embed.
            labels: Keys for the returned dict (one per term).
            cache_path: File path for the ``.pt`` cache.

        Returns:
            Dict mapping each label to its 768-dim embedding tensor.
        """
        cache_path = Path(cache_path)
        if cache_path.exists():
            logger.info("Loading cached embeddings from %s", cache_path)
            return torch.load(cache_path, weights_only=True)

        embeddings = self.embed_terms(terms)
        result = {label: embeddings[i] for i, label in enumerate(labels)}

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, cache_path)
        logger.info("Saved %d embeddings to %s", len(result), cache_path)
        return result


def build_concept_embeddings(
    snomed_mapper: SnomedMapper,
    cache_path: Path = Path("data/processed/concept_embeddings.pt"),
) -> dict[str, torch.Tensor]:
    """Embed all SNOMED concepts from the mapper and cache to disk.

    Iterates over every internal mapping dictionary in the ``SnomedMapper``,
    collects unique ``SnomedConcept`` objects, and produces a 768-dim
    SapBERT embedding for each concept term.

    Args:
        snomed_mapper: A loaded SnomedMapper instance.
        cache_path: Where to save/load the ``.pt`` cache.

    Returns:
        Dict mapping SNOMED code → 768-dim embedding tensor.
    """
    from src.graph_construction.terminology.snomed_mapper import SnomedMapper

    maps = [
        snomed_mapper._icd_map,
        snomed_mapper._labitem_map,
        snomed_mapper._chartitem_map,
        snomed_mapper._drug_map,
        snomed_mapper._organism_map,
        snomed_mapper._comorbidity_map,
        snomed_mapper._loinc_map,
    ]

    concepts: set = set()
    for mapping in maps:
        for entry in mapping.values():
            concept = SnomedMapper._to_concept(entry)
            if concept is not None:
                concepts.add(concept)

    sorted_concepts = sorted(concepts, key=lambda c: c.code)
    terms = [c.term for c in sorted_concepts]
    labels = [c.code for c in sorted_concepts]

    logger.info("Embedding %d unique SNOMED concepts", len(terms))
    return SapBERTEmbedder().embed_and_cache(terms, labels, cache_path)


def embed_unmapped_terms(
    terms: list[str],
    cache_path: Path = Path("data/processed/unmapped_embeddings.pt"),
) -> dict[str, torch.Tensor]:
    """Embed free-text terms that lack SNOMED mappings.

    Useful for clinical terms that could not be mapped to SNOMED-CT
    but still need vector representations for GNN node features.

    Args:
        terms: Raw clinical term strings.
        cache_path: Where to save/load the ``.pt`` cache.

    Returns:
        Dict mapping term string → 768-dim embedding tensor.
    """
    unique_terms = sorted(set(terms))
    logger.info("Embedding %d unique unmapped terms", len(unique_terms))
    return SapBERTEmbedder().embed_and_cache(unique_terms, unique_terms, cache_path)
