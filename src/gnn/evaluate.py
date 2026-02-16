"""GNN evaluation wrapper reusing shared metrics from src.prediction.evaluate."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData

from src.gnn.train import _get_batch_size, _resolve_device
from src.prediction.evaluate import compute_metrics

logger = logging.getLogger(__name__)


def evaluate_gnn(
    model: nn.Module,
    test_loader: Callable[[], object],
    prepare_batch_fn: Callable[[HeteroData], dict],
    output_dir: Path | str | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    """Evaluate a trained GNN model on a test set.

    Parameters
    ----------
    model : nn.Module
        Trained TD4DD model.
    test_loader : Callable
        Factory returning a fresh test-batch iterator.
    prepare_batch_fn : Callable
        Converts a HeteroData batch into model.forward() kwargs.
    output_dir : Path or str, optional
        If provided, saves a JSON metrics report here.
    device : torch.device or str
        Device for inference.

    Returns
    -------
    dict
        Evaluation metrics including auroc, auprc, precision, recall, f1,
        threshold, and confusion_matrix.
    """
    device = _resolve_device(device) if isinstance(device, str) else device
    model = model.to(device)
    model.eval()

    all_probs: list[Tensor] = []
    all_labels: list[Tensor] = []

    with torch.no_grad():
        for batch in test_loader():
            kwargs = prepare_batch_fn(batch)
            out = model(**kwargs)

            batch_size = _get_batch_size(batch)
            labels = batch["admission"].y[:batch_size]
            probs = out["probabilities"].squeeze(-1)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    y_proba = torch.cat(all_probs).numpy()
    y_test = torch.cat(all_labels).numpy()

    nan_mask = np.isnan(y_proba) | np.isinf(y_proba)
    if nan_mask.any():
        n_bad = int(nan_mask.sum())
        logger.warning(
            "NaN/Inf in %d/%d predictions — replacing with 0.5",
            n_bad,
            len(y_proba),
        )
        y_proba = np.where(nan_mask, 0.5, y_proba)

    metrics = compute_metrics(y_proba, y_test)

    # Convert numpy types for JSON serialization
    json_metrics = {}
    for k, v in metrics.items():
        if k == "confusion_matrix":
            json_metrics[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            json_metrics[k] = float(v)
        else:
            json_metrics[k] = v

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        report_path = out_path / "gnn_evaluation.json"
        report_path.write_text(json.dumps(json_metrics, indent=2))
        logger.info("GNN evaluation report saved → %s", report_path)

    return metrics


def _detach_attention_info(info: dict | list | Tensor | None) -> object:
    """Recursively detach and move tensors to CPU."""
    if info is None:
        return None
    if isinstance(info, Tensor):
        return info.detach().cpu()
    if isinstance(info, dict):
        return {k: _detach_attention_info(v) for k, v in info.items()}
    if isinstance(info, (list, tuple)):
        return [_detach_attention_info(v) for v in info]
    return info


def extract_attention_weights(
    model: nn.Module,
    data_loader: Callable[[], object],
    prepare_batch_fn: Callable[[HeteroData], dict],
    top_k: int = 5,
    device: torch.device | str = "cpu",
) -> dict:
    """Extract per-admission attention weights and fusion balance from the model.

    Parameters
    ----------
    model : nn.Module
        Trained TD4DD model.
    data_loader : Callable
        Factory returning a fresh batch iterator.
    prepare_batch_fn : Callable
        Converts a HeteroData batch into model.forward() kwargs.
    top_k : int
        Number of top-attended neighbors to record per admission.
    device : torch.device or str
        Device for inference.

    Returns
    -------
    dict
        Keyed by global admission index with sub-keys:
        - hierarchical_weights: hop-level attention weights (beta_k)
        - top_neighbors: indices with highest attention
        - track_balance: fusion lambda value
    """
    device = _resolve_device(device) if isinstance(device, str) else device
    model = model.to(device)
    model.eval()

    results: dict[int, dict] = {}
    lam = model.lambda_param.detach().cpu().clamp(0.0, 1.0).item()

    with torch.no_grad():
        for batch in data_loader():
            kwargs = prepare_batch_fn(batch)
            out = model(**kwargs)

            batch_size = _get_batch_size(batch)
            attention_info = _detach_attention_info(out.get("attention_info"))

            # Determine global node IDs
            n_id = batch["admission"].n_id if hasattr(batch["admission"], "n_id") else None

            for i in range(batch_size):
                global_idx = n_id[i].item() if n_id is not None else i

                entry: dict = {"track_balance": lam}

                if attention_info is not None and isinstance(attention_info, dict):
                    # Extract hierarchical (hop-level) weights
                    beta = attention_info.get("hierarchical_weights")
                    if beta is not None and isinstance(beta, Tensor):
                        entry["hierarchical_weights"] = beta[i].tolist() if beta.dim() > 1 else beta.tolist()
                    else:
                        entry["hierarchical_weights"] = None

                    # Extract top-k neighbor attention for this admission
                    attn_scores = attention_info.get("attention_scores")
                    if attn_scores is not None and isinstance(attn_scores, Tensor) and attn_scores.dim() >= 2:
                        scores_i = attn_scores[i]
                        k = min(top_k, scores_i.numel())
                        topk_vals, topk_idx = torch.topk(scores_i.flatten(), k)
                        entry["top_neighbors"] = {
                            "indices": topk_idx.tolist(),
                            "scores": topk_vals.tolist(),
                        }
                    else:
                        entry["top_neighbors"] = None
                else:
                    entry["hierarchical_weights"] = None
                    entry["top_neighbors"] = None

                results[global_idx] = entry

    return results
