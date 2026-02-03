"""CLI entry point for feature extraction module.

Usage:
    python -m src.feature_extraction [--input PATH] [--output PATH]
"""

import argparse
import logging
from pathlib import Path

from rdflib import Graph

from src.feature_extraction.feature_builder import build_feature_matrix


logger = logging.getLogger(__name__)


def main():
    """Extract feature matrix from RDF knowledge graph."""
    parser = argparse.ArgumentParser(
        description="Extract ML features from RDF knowledge graph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/processed/knowledge_graph.rdf"),
        help="Path to input RDF graph file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/features/feature_matrix.parquet"),
        help="Path to output parquet file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Check input exists
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    logger.info(f"Loading graph from {args.input}")

    # Load graph
    graph = Graph()
    graph.parse(str(args.input), format="xml")

    logger.info(f"  Loaded {len(graph)} triples")

    # Extract features
    logger.info("Extracting features...")
    feature_df = build_feature_matrix(graph, save_path=args.output)

    print(f"\nFeature extraction complete:")
    print(f"  Shape: {feature_df.shape[0]} admissions x {feature_df.shape[1]} columns")
    print(f"  Output: {args.output}")

    # Show column summary
    print(f"\nColumns:")
    id_cols = ["hadm_id", "subject_id"]
    label_cols = ["readmitted_30d", "readmitted_60d"]
    feature_cols = [c for c in feature_df.columns if c not in id_cols + label_cols]

    print(f"  IDs: {', '.join(id_cols)}")
    print(f"  Labels: {', '.join(label_cols)}")
    print(f"  Features: {len(feature_cols)} columns")

    return 0


if __name__ == "__main__":
    exit(main())
