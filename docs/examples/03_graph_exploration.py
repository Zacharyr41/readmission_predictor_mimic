#!/usr/bin/env python3
"""Graph Exploration Example: SPARQL queries for exploring the knowledge graph.

This example demonstrates how to load and query the RDF knowledge graph
using SPARQL to explore clinical data relationships.

Usage:
    python docs/examples/03_graph_exploration.py

Prerequisites:
    - Run the pipeline first to create the knowledge graph:
      python -m src.main --patients-limit 50 --skip-allen
"""

from pathlib import Path

from rdflib import Graph


def load_graph(path: str = "data/processed/knowledge_graph.rdf") -> Graph:
    """Load the RDF knowledge graph.

    Args:
        path: Path to the RDF file

    Returns:
        Loaded rdflib Graph
    """
    graph_path = Path(path)
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph not found at {path}. "
            "Run the pipeline first: python -m src.main --patients-limit 50 --skip-allen"
        )

    print(f"Loading graph from {path}...")
    graph = Graph()
    graph.parse(str(graph_path))
    print(f"Loaded graph with {len(graph):,} triples\n")
    return graph


def query_basic_stats(graph: Graph) -> None:
    """Print basic graph statistics."""
    print("=" * 60)
    print("Basic Graph Statistics")
    print("=" * 60)

    # Count nodes by type
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT ?type (COUNT(?node) AS ?count)
    WHERE {
        ?node rdf:type ?type .
    }
    GROUP BY ?type
    ORDER BY DESC(?count)
    """

    print("\nNode counts by type:")
    for row in graph.query(query):
        type_name = str(row[0]).split("#")[-1]
        count = int(row[1])
        print(f"  {type_name}: {count:,}")


def query_patient_overview(graph: Graph) -> None:
    """Query patient demographics overview."""
    print("\n" + "=" * 60)
    print("Patient Demographics")
    print("=" * 60)

    query = """
    PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT ?gender (COUNT(?patient) AS ?count) (AVG(?age) AS ?avgAge)
    WHERE {
        ?patient rdf:type mimic:Patient ;
                 mimic:hasGender ?gender ;
                 mimic:hasAge ?age .
    }
    GROUP BY ?gender
    """

    print("\nGender distribution:")
    for row in graph.query(query):
        gender = str(row[0])
        count = int(row[1])
        avg_age = float(row[2])
        print(f"  {gender}: {count} patients, avg age {avg_age:.1f}")


def query_readmission_rates(graph: Graph) -> None:
    """Query readmission rates."""
    print("\n" + "=" * 60)
    print("Readmission Rates")
    print("=" * 60)

    query = """
    PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT
        (COUNT(?admission) AS ?total)
        (SUM(IF(?readmitted30 = true, 1, 0)) AS ?readmitted30d)
        (SUM(IF(?readmitted60 = true, 1, 0)) AS ?readmitted60d)
    WHERE {
        ?admission rdf:type mimic:HospitalAdmission ;
                   mimic:readmittedWithin30Days ?readmitted30 ;
                   mimic:readmittedWithin60Days ?readmitted60 .
    }
    """

    results = list(graph.query(query))
    if results:
        row = results[0]
        total = int(row[0])
        r30 = int(row[1]) if row[1] else 0
        r60 = int(row[2]) if row[2] else 0

        print(f"\nTotal admissions: {total}")
        print(f"30-day readmission rate: {r30}/{total} ({100*r30/total:.1f}%)")
        print(f"60-day readmission rate: {r60}/{total} ({100*r60/total:.1f}%)")


def query_icu_stay_distribution(graph: Graph) -> None:
    """Query ICU stay length distribution."""
    print("\n" + "=" * 60)
    print("ICU Stay Distribution")
    print("=" * 60)

    query = """
    PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>
    PREFIX time: <http://www.w3.org/2006/time#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT ?losDays
    WHERE {
        ?admission rdf:type mimic:HospitalAdmission ;
                   mimic:containsICUStay ?icuStay .
        ?icuStay time:hasDuration ?duration .
        ?duration time:numericDuration ?losDays .
    }
    """

    los_values = [float(row[0]) for row in graph.query(query)]

    if los_values:
        import statistics

        print(f"\nICU stays: {len(los_values)}")
        print(f"Mean LOS: {statistics.mean(los_values):.1f} days")
        print(f"Median LOS: {statistics.median(los_values):.1f} days")
        print(f"Min LOS: {min(los_values):.1f} days")
        print(f"Max LOS: {max(los_values):.1f} days")


def query_event_counts(graph: Graph) -> None:
    """Query clinical event counts by type."""
    print("\n" + "=" * 60)
    print("Clinical Event Counts")
    print("=" * 60)

    event_types = [
        ("BioMarkerEvent", "Lab tests"),
        ("ClinicalSignEvent", "Vital signs"),
        ("MicrobiologyEvent", "Microbiology"),
        ("PrescriptionEvent", "Medications"),
        ("DiagnosisEvent", "Diagnoses"),
    ]

    for event_class, label in event_types:
        query = f"""
        PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT (COUNT(?event) AS ?count)
        WHERE {{
            ?event rdf:type mimic:{event_class} .
        }}
        """

        results = list(graph.query(query))
        count = int(results[0][0]) if results else 0
        print(f"  {label}: {count:,}")


def query_top_diagnoses(graph: Graph, limit: int = 10) -> None:
    """Query most common diagnoses."""
    print("\n" + "=" * 60)
    print(f"Top {limit} Diagnoses")
    print("=" * 60)

    query = f"""
    PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT ?icdCode ?longTitle (COUNT(?diagnosis) AS ?count)
    WHERE {{
        ?diagnosis rdf:type mimic:DiagnosisEvent ;
                   mimic:hasIcdCode ?icdCode .
        OPTIONAL {{ ?diagnosis mimic:hasLongTitle ?longTitle . }}
    }}
    GROUP BY ?icdCode ?longTitle
    ORDER BY DESC(?count)
    LIMIT {limit}
    """

    print(f"\n{'ICD Code':<12} {'Count':<8} {'Description'}")
    print("-" * 60)

    for row in graph.query(query):
        icd = str(row[0])
        title = str(row[1])[:40] if row[1] else "N/A"
        count = int(row[2])
        print(f"{icd:<12} {count:<8} {title}")


def query_temporal_relations(graph: Graph) -> None:
    """Query Allen temporal relation counts."""
    print("\n" + "=" * 60)
    print("Temporal Relations (Allen Algebra)")
    print("=" * 60)

    predicates = [
        ("time:before", "Before"),
        ("time:inside", "During"),
        ("time:intervalOverlaps", "Overlaps"),
        ("time:intervalMeets", "Meets"),
        ("time:intervalStarts", "Starts"),
        ("time:intervalFinishes", "Finishes"),
    ]

    total = 0
    for pred, label in predicates:
        query = f"""
        PREFIX time: <http://www.w3.org/2006/time#>

        SELECT (COUNT(*) AS ?count)
        WHERE {{
            ?a {pred} ?b .
        }}
        """

        results = list(graph.query(query))
        count = int(results[0][0]) if results else 0
        total += count
        print(f"  {label}: {count:,}")

    print(f"\n  Total temporal relations: {total:,}")


def query_sample_patient(graph: Graph) -> None:
    """Query detailed data for a sample patient."""
    print("\n" + "=" * 60)
    print("Sample Patient Detail")
    print("=" * 60)

    # Get first patient
    query_patient = """
    PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT ?patient ?subjectId ?gender ?age
    WHERE {
        ?patient rdf:type mimic:Patient ;
                 mimic:hasSubjectId ?subjectId ;
                 mimic:hasGender ?gender ;
                 mimic:hasAge ?age .
    }
    LIMIT 1
    """

    patient_results = list(graph.query(query_patient))
    if not patient_results:
        print("\nNo patients found in graph.")
        return

    patient = patient_results[0]
    patient_uri = patient[0]
    subject_id = int(patient[1])
    gender = str(patient[2])
    age = int(patient[3])

    print(f"\nPatient: subject_id={subject_id}, gender={gender}, age={age}")

    # Get admissions
    query_admissions = f"""
    PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT ?hadmId ?admissionType ?readmitted30
    WHERE {{
        <{patient_uri}> mimic:hasAdmission ?admission .
        ?admission mimic:hasAdmissionId ?hadmId ;
                   mimic:hasAdmissionType ?admissionType ;
                   mimic:readmittedWithin30Days ?readmitted30 .
    }}
    """

    print("\nAdmissions:")
    for row in graph.query(query_admissions):
        hadm_id = int(row[0])
        adm_type = str(row[1])
        readmitted = str(row[2]).lower() == "true"
        r_str = "Yes" if readmitted else "No"
        print(f"  hadm_id={hadm_id}, type={adm_type}, readmitted_30d={r_str}")


def main():
    """Run all graph exploration queries."""
    try:
        graph = load_graph()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    query_basic_stats(graph)
    query_patient_overview(graph)
    query_readmission_rates(graph)
    query_icu_stay_distribution(graph)
    query_event_counts(graph)
    query_top_diagnoses(graph)
    query_temporal_relations(graph)
    query_sample_patient(graph)

    print("\n" + "=" * 60)
    print("Exploration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
