"""Experiment framework: registry of ablation configurations and runner."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.gnn.evaluate import evaluate_gnn, extract_attention_weights
from src.gnn.hop_extraction import HopExtractor
from src.gnn.losses import LossConfig
from src.gnn.model import ModelConfig, TD4DDModel
from src.gnn.sampling import DualSamplingConfig, DualTrackSampler, build_auxiliary_subgraphs
from src.gnn.train import Trainer, TrainingConfig
from src.gnn.view_adapter import GraphViewAdapter, GraphViewConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _set_seed(seed: int) -> None:
    """Set seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _infer_feat_dims(view_data: HeteroData) -> dict[str, int]:
    """Infer feature dimensions from a HeteroData object."""
    dims: dict[str, int] = {}
    for ntype in view_data.node_types:
        store = view_data[ntype]
        if hasattr(store, "x") and store.x is not None:
            dims[ntype] = store.x.shape[1]
    return dims


def _auto_meta_paths(
    view_data: HeteroData, view_config: GraphViewConfig
) -> list[list[str]]:
    """Generate 2-hop meta-paths from view edges for diffusion aux subgraphs.

    For each entity type in active_entity_types, creates a meta-path
    [forward_rel, reverse_rel] through that entity.
    """
    target = view_config.target_node_type
    meta_paths: list[list[str]] = []

    for et in view_data.edge_types:
        src, rel, dst = et
        if src == target and dst != target and not rel.startswith("rev_"):
            rev_rel = f"rev_{rel}"
            rev_key = (dst, rev_rel, target)
            if rev_key in view_data.edge_types:
                meta_paths.append([rel, rev_rel])

    return meta_paths


def _convert_paths(d: Any) -> Any:
    """Recursively convert Path objects to strings for JSON serialization."""
    if isinstance(d, dict):
        return {k: _convert_paths(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_convert_paths(v) for v in d]
    if isinstance(d, tuple):
        return [_convert_paths(v) for v in d]
    if isinstance(d, Path):
        return str(d)
    return d


def _build_contextual_edge_sequences(
    view_data: HeteroData,
    config: ExperimentConfig,
) -> list[list[tuple[str, str, str]]]:
    """Derive alternating edge sequences for the contextual transformer paths.

    One sequence per active entity type (up to ``num_contextual_paths``).
    Each sequence alternates: ``[target→entity, entity→target, ...]`` for
    ``k_hops_contextual`` hops.
    """
    target = config.view_config.target_node_type
    sequences: list[list[tuple[str, str, str]]] = []

    # Find forward edges from target to entity types
    for et in view_data.edge_types:
        src, rel, dst = et
        if src == target and dst != target and not rel.startswith("rev_"):
            rev_rel = f"rev_{rel}"
            rev_key = (dst, rev_rel, target)
            if rev_key in view_data.edge_types:
                # Build alternating sequence for k_hops_contextual hops
                seq: list[tuple[str, str, str]] = []
                for k in range(config.k_hops_contextual):
                    if k % 2 == 0:
                        seq.append(et)  # target → entity
                    else:
                        seq.append(rev_key)  # entity → target
                sequences.append(seq)

        if len(sequences) >= config.num_contextual_paths:
            break

    return sequences


def _serialize_metrics(metrics: dict) -> dict:
    """Convert numpy types to Python types for JSON serialization."""
    result = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            result[k] = float(v)
        elif isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif isinstance(v, dict):
            result[k] = _serialize_metrics(v)
        else:
            result[k] = v
    return result


# ──────────────────────────────────────────────────────────────────────────────
# ExperimentConfig
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ExperimentConfig:
    """Configuration for a single ablation experiment."""

    name: str = ""
    description: str = ""

    # Model toggles
    use_transformer: bool = True
    use_diffusion: bool = True
    use_temporal_encoding: bool = True

    # Architecture params
    d_model: int = 128
    num_contextual_paths: int = 2
    k_hops_contextual: int = 4
    k_hops_temporal: int = 2
    nhead: int = 4
    dropout: float = 0.3

    # Sub-configs
    view_config: GraphViewConfig = field(default_factory=GraphViewConfig.readmission_default)
    loss_config: LossConfig = field(default_factory=LossConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    sampling_config: DualSamplingConfig | None = None

    def to_model_config(self, feat_dims: dict[str, int]) -> ModelConfig:
        """Map ExperimentConfig fields to ModelConfig, injecting runtime feat_dims."""
        return ModelConfig(
            feat_dims=feat_dims,
            d_model=self.d_model,
            num_contextual_paths=self.num_contextual_paths,
            k_hops_contextual=self.k_hops_contextual,
            k_hops_temporal=self.k_hops_temporal,
            nhead=self.nhead,
            dropout=self.dropout,
            use_transformer=self.use_transformer,
            use_diffusion=self.use_diffusion,
            use_temporal_encoding=self.use_temporal_encoding,
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        d = asdict(self)
        return _convert_paths(d)

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentConfig:
        """Reconstruct from a dict (inverse of to_dict)."""
        d = dict(d)  # shallow copy

        # Reconstruct nested dataclass sub-configs
        if "view_config" in d and isinstance(d["view_config"], dict):
            d["view_config"] = GraphViewConfig(**d["view_config"])
        if "loss_config" in d and isinstance(d["loss_config"], dict):
            d["loss_config"] = LossConfig(**d["loss_config"])
        if "training_config" in d and isinstance(d["training_config"], dict):
            tc = d["training_config"]
            if "checkpoint_dir" in tc:
                tc["checkpoint_dir"] = Path(tc["checkpoint_dir"])
            d["training_config"] = TrainingConfig(**tc)
        if "sampling_config" in d and d["sampling_config"] is not None:
            # DualSamplingConfig has nested SamplingTrackConfig fields;
            # for simplicity, leave as None and let runner derive it.
            d["sampling_config"] = None

        # Filter to only valid fields
        valid_fields = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in valid_fields}

        return cls(**d)


# ──────────────────────────────────────────────────────────────────────────────
# Experiment Registry
# ──────────────────────────────────────────────────────────────────────────────


EXPERIMENT_REGISTRY: dict[str, ExperimentConfig] = {
    "E1_mlp_baseline": ExperimentConfig(
        name="E1_mlp_baseline",
        description="Floor performance baseline: projection + classifier only",
        use_transformer=False,
        use_diffusion=False,
        use_temporal_encoding=False,
    ),
    "E2_mlp_sapbert": ExperimentConfig(
        name="E2_mlp_sapbert",
        description="Value of SapBERT priors: MLP with enriched features",
        use_transformer=False,
        use_diffusion=False,
        use_temporal_encoding=False,
    ),
    "E3_transformer_only": ExperimentConfig(
        name="E3_transformer_only",
        description="Decisive: does graph structure help over MLP baseline?",
        use_transformer=True,
        use_diffusion=False,
        use_temporal_encoding=False,
    ),
    "E4_transformer_temporal": ExperimentConfig(
        name="E4_transformer_temporal",
        description="Value of temporal encoding on top of transformer",
        use_transformer=True,
        use_diffusion=False,
        use_temporal_encoding=True,
    ),
    "E5_full_no_temporal": ExperimentConfig(
        name="E5_full_no_temporal",
        description="Value of cross-view denoising (diffusion without temporal)",
        use_transformer=True,
        use_diffusion=True,
        use_temporal_encoding=False,
    ),
    "E6_full_model": ExperimentConfig(
        name="E6_full_model",
        description="Complete TD4DD model with all components",
        use_transformer=True,
        use_diffusion=True,
        use_temporal_encoding=True,
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# ExperimentRunner
# ──────────────────────────────────────────────────────────────────────────────


class ExperimentRunner:
    """Orchestrates end-to-end experiment runs.

    Parameters
    ----------
    graph_path : Path
        Path to a saved HeteroData .pt file.
    """

    def __init__(
        self,
        graph_path: Path,
        base_output_dir: Path | None = None,
    ) -> None:
        self.graph_path = Path(graph_path)
        self.base_output_dir = Path(base_output_dir) if base_output_dir else Path("outputs/gnn_experiments")
        self._cached_data: HeteroData | None = None

    def _load_data(self) -> HeteroData:
        """Load and cache the full HeteroData graph."""
        if self._cached_data is None:
            logger.info("Loading graph from %s", self.graph_path)
            self._cached_data = torch.load(
                self.graph_path, map_location="cpu", weights_only=False
            )
        return self._cached_data

    def run(
        self,
        experiment_name: str,
        seed: int = 42,
        config_overrides: dict | None = None,
    ) -> dict:
        """Run a single experiment end-to-end.

        Parameters
        ----------
        experiment_name : str
            Key in EXPERIMENT_REGISTRY.
        seed : int
            Random seed for reproducibility.
        config_overrides : dict, optional
            Overrides merged into the experiment config.

        Returns
        -------
        dict
            Keys: experiment_name, config, train_result, eval_metrics,
            elapsed_seconds, output_dir.
        """
        start = time.time()

        # 1. Config resolution
        if experiment_name not in EXPERIMENT_REGISTRY:
            raise KeyError(
                f"Unknown experiment: {experiment_name!r}. "
                f"Available: {list(EXPERIMENT_REGISTRY.keys())}"
            )
        config = EXPERIMENT_REGISTRY[experiment_name]
        if config_overrides:
            d = config.to_dict()
            d.update(config_overrides)
            config = ExperimentConfig.from_dict(d)
            config.name = experiment_name

        # 2. Set seeds
        _set_seed(seed)

        # 3. Load data
        full_data = self._load_data()

        # 4. Apply view
        adapter = GraphViewAdapter(config.view_config)
        view_data = adapter.apply(full_data)

        # 5. Build auxiliary subgraphs (if diffusion enabled)
        aux_edge_indices = None
        if config.use_diffusion:
            meta_paths = config.view_config.meta_paths
            if not meta_paths:
                meta_paths = _auto_meta_paths(view_data, config.view_config)
            if meta_paths:
                aux_edge_indices = build_auxiliary_subgraphs(
                    view_data,
                    config.view_config.target_node_type,
                    meta_paths,
                )

        # 6. Create sampler
        sampling_config = config.sampling_config
        if sampling_config is None:
            sampling_config = DualSamplingConfig.from_view_config(
                config.view_config, view_data
            )
        sampler = DualTrackSampler(
            view_data, sampling_config, config.training_config.batch_size
        )

        # 7. Infer feat_dims and resolve contextual paths
        feat_dims = _infer_feat_dims(view_data)
        edge_sequences: list[list[tuple[str, str, str]]] = []
        if config.use_transformer:
            edge_sequences = _build_contextual_edge_sequences(view_data, config)

        # 8. Create model (adjust num_contextual_paths to actual count)
        model_config = config.to_model_config(feat_dims)
        if edge_sequences:
            model_config.num_contextual_paths = len(edge_sequences)
        model = TD4DDModel(model_config)

        # 9. Configure output paths
        output_dir = self.base_output_dir / experiment_name
        training_config = TrainingConfig(
            lr=config.training_config.lr,
            batch_size=config.training_config.batch_size,
            max_epochs=config.training_config.max_epochs,
            patience=config.training_config.patience,
            weight_decay=config.training_config.weight_decay,
            grad_clip_norm=config.training_config.grad_clip_norm,
            device=config.training_config.device,
            checkpoint_dir=output_dir / "checkpoints",
            log_every_n_epochs=config.training_config.log_every_n_epochs,
        )

        # 10. Construct HopExtractor if transformer is enabled
        hop_extractor = None
        if config.use_transformer and edge_sequences:
            temporal_types = sampling_config.temporal.edge_types if sampling_config.temporal.edge_types else []
            hop_extractor = HopExtractor(
                contextual_edge_sequences=edge_sequences,
                temporal_edge_types=temporal_types,
                k_hops_contextual=config.k_hops_contextual,
                k_hops_temporal=config.k_hops_temporal if config.use_temporal_encoding else 0,
                neighbors_per_hop_contextual=sampling_config.contextual.neighbors_per_hop,
                neighbors_per_hop_temporal=sampling_config.temporal.neighbors_per_hop,
            )

        # 11. Create Trainer
        trainer = Trainer(
            model=model,
            train_loader=sampler.get_train_loader,
            val_loader=sampler.get_val_loader,
            loss_config=config.loss_config,
            training_config=training_config,
            aux_edge_indices=aux_edge_indices,
            hop_extractor=hop_extractor,
        )

        # 12. Train
        train_result = trainer.train()

        # 13. Evaluate
        eval_metrics = evaluate_gnn(
            model=model,
            test_loader=sampler.get_test_loader,
            prepare_batch_fn=trainer._prepare_batch,
            output_dir=output_dir,
            device=training_config.device,
        )

        # 14. Save results
        output_dir.mkdir(parents=True, exist_ok=True)

        # metrics.json
        (output_dir / "metrics.json").write_text(
            json.dumps(_serialize_metrics(eval_metrics), indent=2)
        )

        # config.json
        (output_dir / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2)
        )

        # training_history.json
        (output_dir / "training_history.json").write_text(
            json.dumps(train_result["training_history"], indent=2)
        )

        # attention_weights.pt (if transformer enabled)
        if config.use_transformer:
            try:
                attn = extract_attention_weights(
                    model=model,
                    data_loader=sampler.get_test_loader,
                    prepare_batch_fn=trainer._prepare_batch,
                    device=training_config.device,
                )
                torch.save(attn, output_dir / "attention_weights.pt")
            except Exception:
                logger.warning("Could not extract attention weights", exc_info=True)

        elapsed = time.time() - start
        logger.info(
            "Experiment %s completed in %.1fs — AUROC=%.4f AUPRC=%.4f",
            experiment_name,
            elapsed,
            eval_metrics.get("auroc", 0.0),
            eval_metrics.get("auprc", 0.0),
        )

        return {
            "experiment_name": experiment_name,
            "config": config.to_dict(),
            "train_result": {
                "best_val_auroc": train_result["best_val_auroc"],
                "best_epoch": train_result["best_epoch"],
            },
            "eval_metrics": _serialize_metrics(eval_metrics),
            "elapsed_seconds": elapsed,
            "output_dir": str(output_dir),
        }

    def run_all(
        self,
        experiments: list[str] | None = None,
        seed: int = 42,
    ) -> dict[str, dict]:
        """Run multiple experiments sequentially.

        Parameters
        ----------
        experiments : list[str], optional
            Experiment names to run. Defaults to all registered.
        seed : int
            Random seed.

        Returns
        -------
        dict
            Mapping from experiment name to result dict.
        """
        names = experiments or list(EXPERIMENT_REGISTRY.keys())
        results: dict[str, dict] = {}
        for name in names:
            logger.info("Starting experiment: %s", name)
            results[name] = self.run(name, seed=seed)
        return results

    @staticmethod
    def compare(results: dict[str, dict]) -> str:
        """Generate a markdown comparison table from experiment results.

        Parameters
        ----------
        results : dict
            Mapping from experiment name to result dict (as returned by run).

        Returns
        -------
        str
            Markdown-formatted table.
        """
        lines = [
            "| Experiment | AUROC | AUPRC | F1 | Best Epoch | Time (s) |",
            "|------------|-------|-------|----|------------|----------|",
        ]
        for name, res in results.items():
            em = res.get("eval_metrics", {})
            tr = res.get("train_result", {})
            auroc = em.get("auroc", "N/A")
            auprc = em.get("auprc", "N/A")
            f1 = em.get("f1", "N/A")
            best_epoch = tr.get("best_epoch", "N/A")
            elapsed = res.get("elapsed_seconds", "N/A")

            auroc_str = f"{auroc:.4f}" if isinstance(auroc, (int, float)) else str(auroc)
            auprc_str = f"{auprc:.4f}" if isinstance(auprc, (int, float)) else str(auprc)
            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
            elapsed_str = f"{elapsed:.1f}" if isinstance(elapsed, (int, float)) else str(elapsed)

            lines.append(
                f"| {name} | {auroc_str} | {auprc_str} | {f1_str} | {best_epoch} | {elapsed_str} |"
            )

        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for experiment execution."""
    parser = argparse.ArgumentParser(description="GNN Experiment Runner")
    parser.add_argument("--run", type=str, help="Run a specific experiment by name")
    parser.add_argument("--run-all", action="store_true", help="Run all experiments")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument(
        "--graph", type=str, default="data/processed/full_hetero_graph.pt",
        help="Path to the HeteroData .pt file",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for name, config in EXPERIMENT_REGISTRY.items():
            print(f"  {name}: {config.description}")
        return

    if not args.run and not args.run_all:
        parser.print_help()
        return

    runner = ExperimentRunner(Path(args.graph))

    if args.run:
        result = runner.run(args.run, seed=args.seed)
        print(f"\nResult: AUROC={result['eval_metrics'].get('auroc', 'N/A')}")

    if args.run_all:
        results = runner.run_all(seed=args.seed)
        print("\n" + runner.compare(results))


if __name__ == "__main__":
    main()
