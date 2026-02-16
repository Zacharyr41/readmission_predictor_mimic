"""CLI entry point for graph analysis module.

Usage:
    python -m src.graph_analysis [--input PATH] [--output PATH]
"""

import argparse
import logging
from pathlib import Path

from rdflib import Graph

from src.graph_analysis.analysis import generate_analysis_report


logger = logging.getLogger(__name__)


def main():
    """Generate graph structure analysis report."""
    parser = argparse.ArgumentParser(
        description="Analyze RDF knowledge graph structure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/processed/knowledge_graph.nt"),
        help="Path to input RDF graph file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("outputs/reports/graph_analysis.md"),
        help="Path to output markdown report",
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

    # Load graph into disk-backed store
    from src.graph_construction.disk_graph import (
        bind_namespaces,
        close_disk_graph,
        open_disk_graph,
    )

    store_path = args.input.parent / "oxigraph_analysis_store"
    graph = open_disk_graph(store_path)
    bind_namespaces(graph)
    graph.parse(str(args.input), format="nt")

    logger.info(f"  Loaded {len(graph)} triples")

    # Generate report
    logger.info(f"Generating analysis report...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report, _nx_graph = generate_analysis_report(graph, output_path=args.output)

    close_disk_graph(graph)

    print(f"\nAnalysis report generated:")
    print(f"  Output: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
