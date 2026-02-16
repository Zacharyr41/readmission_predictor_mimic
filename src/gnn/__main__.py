"""GNN pipeline orchestrator.

Chains RDF knowledge graph → SapBERT embeddings → HeteroData export →
experiment execution. Run with ``python -m src.gnn --help`` for usage.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Default paths
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_RDF = Path("data/processed/knowledge_graph.nt")
DEFAULT_FEATURES = Path("data/features/feature_matrix.parquet")
DEFAULT_EMBEDDINGS = Path("data/processed/concept_embeddings.pt")
DEFAULT_OUTPUT = Path("data/processed/full_hetero_graph.pt")
DEFAULT_MAPPINGS = Path("data/mappings")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline functions
# ──────────────────────────────────────────────────────────────────────────────


def build_embeddings(
    mappings_dir: Path = DEFAULT_MAPPINGS,
    embeddings_path: Path = DEFAULT_EMBEDDINGS,
) -> None:
    """Generate SapBERT concept embeddings from SNOMED mappings."""
    from src.gnn.embeddings import build_concept_embeddings
    from src.graph_construction.terminology.snomed_mapper import SnomedMapper

    logger.info("Loading SNOMED mappings from %s", mappings_dir)
    mapper = SnomedMapper(mappings_dir)

    logger.info("Building concept embeddings → %s", embeddings_path)
    result = build_concept_embeddings(mapper, cache_path=embeddings_path)
    logger.info("Embedded %d concepts", len(result))


def _make_split_fn():
    """Create a split function matching graph_export's expected signature."""
    from src.prediction.split import patient_level_split

    def split_fn(df, target_col):
        return patient_level_split(df, target_col, subject_col="subject_id")

    return split_fn


def export_graph(
    rdf_path: Path = DEFAULT_RDF,
    features_path: Path = DEFAULT_FEATURES,
    embeddings_path: Path = DEFAULT_EMBEDDINGS,
    output_path: Path = DEFAULT_OUTPUT,
) -> None:
    """Convert RDF knowledge graph to PyG HeteroData."""
    from src.gnn.graph_export import export_rdf_to_heterodata
    from src.graph_construction.disk_graph import (
        bind_namespaces,
        close_disk_graph,
        open_disk_graph,
    )

    logger.info("Parsing RDF graph from %s", rdf_path)
    store_path = rdf_path.parent / "oxigraph_export_store"
    rdf_graph = open_disk_graph(store_path)
    bind_namespaces(rdf_graph)
    rdf_graph.parse(str(rdf_path), format="nt")
    logger.info("RDF graph loaded: %d triples", len(rdf_graph))

    split_fn = _make_split_fn()

    logger.info("Exporting HeteroData → %s", output_path)
    data = export_rdf_to_heterodata(
        rdf_graph,
        features_path,
        embeddings_path,
        split_fn=split_fn,
        output_path=output_path,
        embed_unmapped_fn=None,
    )
    close_disk_graph(rdf_graph)
    logger.info(
        "HeteroData saved: %d node types, %d edge types",
        len(data.node_types),
        len(data.edge_types),
    )


def prepare(
    rdf_path: Path = DEFAULT_RDF,
    features_path: Path = DEFAULT_FEATURES,
    embeddings_path: Path = DEFAULT_EMBEDDINGS,
    output_path: Path = DEFAULT_OUTPUT,
    mappings_dir: Path = DEFAULT_MAPPINGS,
) -> None:
    """Run both build_embeddings and export_graph in sequence."""
    build_embeddings(mappings_dir=mappings_dir, embeddings_path=embeddings_path)
    export_graph(
        rdf_path=rdf_path,
        features_path=features_path,
        embeddings_path=embeddings_path,
        output_path=output_path,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the GNN pipeline."""
    parser = argparse.ArgumentParser(
        prog="python -m src.gnn",
        description="GNN pipeline: embeddings → graph export → experiments",
    )

    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--build-embeddings",
        action="store_true",
        help="Build SapBERT concept embeddings from SNOMED mappings",
    )
    actions.add_argument(
        "--export-graph",
        action="store_true",
        help="Export RDF knowledge graph to PyG HeteroData",
    )
    actions.add_argument(
        "--prepare",
        action="store_true",
        help="Run both --build-embeddings and --export-graph",
    )
    actions.add_argument(
        "--run",
        type=str,
        metavar="EXPERIMENT",
        help="Run a specific experiment by name",
    )
    actions.add_argument(
        "--run-all",
        action="store_true",
        help="Run all registered ablation experiments",
    )
    actions.add_argument(
        "--list",
        action="store_true",
        help="List available experiments",
    )

    parser.add_argument(
        "--rdf", type=Path, default=DEFAULT_RDF, help="Path to RDF file"
    )
    parser.add_argument(
        "--features", type=Path, default=DEFAULT_FEATURES, help="Path to feature matrix"
    )
    parser.add_argument(
        "--embeddings", type=Path, default=DEFAULT_EMBEDDINGS, help="Path to embeddings .pt"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Path to HeteroData .pt"
    )
    parser.add_argument(
        "--mappings", type=Path, default=DEFAULT_MAPPINGS, help="SNOMED mappings directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging"
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.build_embeddings:
        build_embeddings(
            mappings_dir=args.mappings,
            embeddings_path=args.embeddings,
        )
    elif args.export_graph:
        export_graph(
            rdf_path=args.rdf,
            features_path=args.features,
            embeddings_path=args.embeddings,
            output_path=args.output,
        )
    elif args.prepare:
        prepare(
            rdf_path=args.rdf,
            features_path=args.features,
            embeddings_path=args.embeddings,
            output_path=args.output,
            mappings_dir=args.mappings,
        )
    elif args.run:
        from src.gnn.experiments import ExperimentRunner

        runner = ExperimentRunner(args.output)
        result = runner.run(args.run, seed=args.seed)
        print(
            f"\n{args.run}: AUROC={result['eval_metrics'].get('auroc', 'N/A')}"
        )
    elif args.run_all:
        from src.gnn.experiments import ExperimentRunner

        runner = ExperimentRunner(args.output)
        results = runner.run_all(seed=args.seed)
        print("\n" + runner.compare(results))
    elif args.list:
        from src.gnn.experiments import EXPERIMENT_REGISTRY

        print("Available experiments:")
        for name, config in EXPERIMENT_REGISTRY.items():
            print(f"  {name}: {config.description}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
