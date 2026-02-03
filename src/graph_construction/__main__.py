"""CLI entry point for graph construction module.

Usage:
    python -m src.graph_construction [--db-path PATH] [--output PATH] [--patients-limit N]
"""

import argparse
import logging
from pathlib import Path

from config.settings import Settings
from src.graph_construction.pipeline import build_graph


logger = logging.getLogger(__name__)


# Default ontology directory
DEFAULT_ONTOLOGY_DIR = Path(__file__).parent.parent.parent / "ontology" / "definition"


def main():
    """Build RDF knowledge graph from DuckDB."""
    parser = argparse.ArgumentParser(
        description="Build RDF knowledge graph from MIMIC-IV DuckDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--db-path",
        type=Path,
        help="Path to DuckDB file (default: from settings)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/processed/knowledge_graph.rdf"),
        help="Path to output RDF file",
    )
    parser.add_argument(
        "--ontology-dir",
        type=Path,
        default=DEFAULT_ONTOLOGY_DIR,
        help="Path to ontology definition files",
    )
    parser.add_argument(
        "--icd-codes",
        nargs="+",
        default=["I63", "I61", "I60"],
        help="ICD-10 code prefixes for cohort selection",
    )
    parser.add_argument(
        "--patients-limit",
        type=int,
        default=0,
        help="Maximum patients to process (0 = no limit)",
    )
    parser.add_argument(
        "--biomarkers-limit",
        type=int,
        default=0,
        help="Maximum biomarkers per ICU stay (0 = no limit)",
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

    # Load settings
    settings = Settings()
    db_path = args.db_path or settings.duckdb_path

    logger.info(f"Building graph from {db_path}")
    logger.info(f"Output: {args.output}")
    logger.info(f"ICD codes: {args.icd_codes}")

    # Build graph
    graph = build_graph(
        db_path=db_path,
        ontology_dir=args.ontology_dir,
        output_path=args.output,
        icd_prefixes=args.icd_codes,
        patients_limit=args.patients_limit,
        biomarkers_limit=args.biomarkers_limit,
    )

    print(f"\nGraph built successfully:")
    print(f"  Triples: {len(graph)}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
