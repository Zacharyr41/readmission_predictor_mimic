"""Neo4j integration for RDF graph import and Cypher queries.

This module provides functions to import RDF graphs into Neo4j using the n10s
(neosemantics) plugin and execute Cypher queries.
"""

from pathlib import Path
from typing import Any


def check_neo4j_connection(uri: str, user: str, password: str) -> bool:
    """Check if Neo4j is reachable.

    Args:
        uri: Neo4j bolt URI (e.g., "bolt://localhost:7687").
        user: Neo4j username.
        password: Neo4j password.

    Returns:
        True if connection successful, False otherwise.
    """
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        return True
    except ImportError:
        return False
    except Exception:
        return False


def import_rdf_to_neo4j(
    rdf_path: Path,
    neo4j_uri: str,
    user: str,
    password: str,
    graph_config: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Import RDF file into Neo4j using n10s (neosemantics) plugin.

    Requires the n10s plugin to be installed in Neo4j.
    See: https://neo4j.com/labs/neosemantics/

    Args:
        rdf_path: Path to RDF file (TTL, RDF/XML, etc.).
        neo4j_uri: Neo4j bolt URI.
        user: Neo4j username.
        password: Neo4j password.
        graph_config: Optional n10s graph configuration.

    Returns:
        Dictionary with import statistics:
        - triples_loaded: Number of triples imported
        - nodes_created: Number of nodes created
        - relationships_created: Number of relationships created
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))

    try:
        with driver.session() as session:
            # Initialize n10s if not already done
            try:
                session.run("CALL n10s.graphconfig.init()")
            except Exception:
                # Config may already exist
                pass

            # Set graph config if provided
            if graph_config:
                session.run("CALL n10s.graphconfig.set($config)", config=graph_config)

            # Import RDF file
            file_uri = f"file://{rdf_path.absolute()}"

            # Determine format from extension
            suffix = rdf_path.suffix.lower()
            format_map = {
                ".ttl": "Turtle",
                ".turtle": "Turtle",
                ".rdf": "RDF/XML",
                ".xml": "RDF/XML",
                ".nt": "N-Triples",
                ".n3": "N3",
                ".jsonld": "JSON-LD",
            }
            rdf_format = format_map.get(suffix, "Turtle")

            result = session.run(
                "CALL n10s.rdf.import.fetch($uri, $format)",
                uri=file_uri,
                format=rdf_format,
            )

            record = result.single()
            if record:
                return {
                    "triples_loaded": record.get("triplesLoaded", 0),
                    "nodes_created": record.get("nodesCreated", 0),
                    "relationships_created": record.get("relationshipsCreated", 0),
                }

            return {"triples_loaded": 0, "nodes_created": 0, "relationships_created": 0}

    finally:
        driver.close()


def query_neo4j(
    cypher: str,
    neo4j_uri: str,
    user: str,
    password: str,
    parameters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Execute a Cypher query and return results.

    Args:
        cypher: Cypher query string.
        neo4j_uri: Neo4j bolt URI.
        user: Neo4j username.
        password: Neo4j password.
        parameters: Optional query parameters.

    Returns:
        List of result dictionaries.
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))

    try:
        with driver.session() as session:
            result = session.run(cypher, parameters or {})
            return [dict(record) for record in result]

    finally:
        driver.close()


def clear_neo4j_database(neo4j_uri: str, user: str, password: str) -> None:
    """Clear all data from Neo4j database.

    Args:
        neo4j_uri: Neo4j bolt URI.
        user: Neo4j username.
        password: Neo4j password.
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))

    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    finally:
        driver.close()
