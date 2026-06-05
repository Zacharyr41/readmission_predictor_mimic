"""Tests for the conversational graph builder module."""

from datetime import datetime
from pathlib import Path

import pytest
from rdflib.namespace import RDF, XSD

from src.conversational.models import ExtractionResult
from src.conversational.graph_builder import build_query_graph
from src.graph_construction.ontology import MIMIC_NS, TIME_NS
from src.graph_construction.terminology.drug_category import DrugCategoryResolver

_MAPPINGS_DIR = Path(__file__).resolve().parents[2] / "data" / "mappings"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ONTOLOGY_DIR = Path(__file__).parent.parent.parent / "ontology" / "definition"


@pytest.fixture
def ontology_dir():
    return ONTOLOGY_DIR


@pytest.fixture
def empty_extraction():
    return ExtractionResult()


@pytest.fixture
def minimal_extraction():
    """1 patient, 1 admission, 1 ICU stay (3 days), 1 event per type.

    Biomarker has NO stay_id key (matching real extractor output).
    Vital HAS stay_id (matching chartevents extractor output).
    """
    return ExtractionResult(
        patients=[{"subject_id": 9001, "gender": "M", "anchor_age": 60}],
        admissions=[{
            "hadm_id": 9101, "subject_id": 9001,
            "admittime": datetime(2150, 6, 1, 8, 0),
            "dischtime": datetime(2150, 6, 10, 14, 0),
            "admission_type": "EMERGENCY",
            "discharge_location": "HOME",
        }],
        icu_stays=[{
            "stay_id": 9201, "hadm_id": 9101, "subject_id": 9001,
            "intime": datetime(2150, 6, 1, 10, 0),
            "outtime": datetime(2150, 6, 4, 10, 0),
            "los": 3.0,
        }],
        events={
            "biomarker": [{
                "labevent_id": 90001, "subject_id": 9001, "hadm_id": 9101,
                "itemid": 50912, "charttime": datetime(2150, 6, 2, 6, 0),
                "label": "Creatinine", "fluid": "Blood", "category": "Chemistry",
                "valuenum": 1.1, "valueuom": "mg/dL",
                "ref_range_lower": 0.7, "ref_range_upper": 1.3,
            }],
            "vital": [{
                "stay_id": 9201, "subject_id": 9001, "hadm_id": 9101,
                "itemid": 220045, "charttime": datetime(2150, 6, 1, 14, 0),
                "label": "Heart Rate", "category": "Routine Vital Signs",
                "valuenum": 82.0,
            }],
            "drug": [{
                "hadm_id": 9101, "subject_id": 9001,
                "drug": "Vancomycin", "starttime": datetime(2150, 6, 1, 12, 0),
                "stoptime": datetime(2150, 6, 3, 12, 0),
                "dose_val_rx": 1000.0, "dose_unit_rx": "mg", "route": "IV",
            }],
            "diagnosis": [{
                "hadm_id": 9101, "subject_id": 9001, "seq_num": 1,
                "icd_code": "I63.0", "icd_version": 10,
                "long_title": "Cerebral infarction due to thrombosis",
            }],
            "microbiology": [{
                "microevent_id": 95001, "subject_id": 9001, "hadm_id": 9101,
                "charttime": datetime(2150, 6, 2, 12, 0),
                "spec_type_desc": "BLOOD CULTURE",
                "org_name": "STAPHYLOCOCCUS AUREUS",
            }],
        },
    )


@pytest.fixture
def multi_patient_extraction():
    """2 patients. Patient 1 has 2 sequential admissions and 2 ICU stays."""
    return ExtractionResult(
        patients=[
            {"subject_id": 8001, "gender": "F", "anchor_age": 72},
            {"subject_id": 8002, "gender": "M", "anchor_age": 55},
        ],
        admissions=[
            {
                "hadm_id": 8101, "subject_id": 8001,
                "admittime": datetime(2150, 3, 1, 8, 0),
                "dischtime": datetime(2150, 3, 10, 14, 0),
                "admission_type": "EMERGENCY",
                "discharge_location": "HOME",
            },
            {
                "hadm_id": 8102, "subject_id": 8001,
                "admittime": datetime(2150, 4, 5, 10, 0),
                "dischtime": datetime(2150, 4, 12, 16, 0),
                "admission_type": "URGENT",
                "discharge_location": "SNF",
            },
            {
                "hadm_id": 8103, "subject_id": 8002,
                "admittime": datetime(2150, 5, 1, 6, 0),
                "dischtime": datetime(2150, 5, 8, 12, 0),
                "admission_type": "ELECTIVE",
                "discharge_location": "HOME",
            },
        ],
        icu_stays=[
            {
                "stay_id": 8201, "hadm_id": 8101, "subject_id": 8001,
                "intime": datetime(2150, 3, 1, 10, 0),
                "outtime": datetime(2150, 3, 4, 10, 0),
                "los": 3.0,
            },
            {
                "stay_id": 8202, "hadm_id": 8103, "subject_id": 8002,
                "intime": datetime(2150, 5, 1, 8, 0),
                "outtime": datetime(2150, 5, 3, 8, 0),
                "los": 2.0,
            },
        ],
        events={},
    )


@pytest.fixture
def temporal_cohort_extraction():
    """A patient whose ICU stay carries an adjacency (meets) + precedence
    (before) chain — the substrate the temporal cohort path (plan III-A)
    queries for dose-trend traits like "escalating vasopressors".

    Two vasopressor administrations *meet* (the first ends exactly when the
    second begins); a later drug starts strictly after both, giving coarse
    precedence. All three are proper intervals, so the Allen classifier emits
    ``time:intervalMeets`` for the adjacency and ``time:before`` for precedence.
    """
    return ExtractionResult(
        patients=[{"subject_id": 6001, "gender": "F", "anchor_age": 68}],
        admissions=[{
            "hadm_id": 6101, "subject_id": 6001,
            "admittime": datetime(2150, 6, 1, 8, 0),
            "dischtime": datetime(2150, 6, 5, 14, 0),
            "admission_type": "EMERGENCY",
            "discharge_location": "HOME",
        }],
        icu_stays=[{
            "stay_id": 6201, "hadm_id": 6101, "subject_id": 6001,
            "intime": datetime(2150, 6, 1, 10, 0),
            "outtime": datetime(2150, 6, 4, 10, 0),
            "los": 3.0,
        }],
        events={
            "drug": [
                {
                    "hadm_id": 6101, "subject_id": 6001,
                    "drug": "Norepinephrine",
                    "starttime": datetime(2150, 6, 1, 12, 0),
                    "stoptime": datetime(2150, 6, 1, 18, 0),
                    "dose_val_rx": 5.0, "dose_unit_rx": "mcg/kg/min", "route": "IV",
                },
                {
                    "hadm_id": 6101, "subject_id": 6001,
                    "drug": "Vasopressin",
                    "starttime": datetime(2150, 6, 1, 18, 0),  # meets Norepinephrine
                    "stoptime": datetime(2150, 6, 2, 0, 0),
                    "dose_val_rx": 0.04, "dose_unit_rx": "units/min", "route": "IV",
                },
                {
                    "hadm_id": 6101, "subject_id": 6001,
                    "drug": "Phenylephrine",
                    "starttime": datetime(2150, 6, 3, 6, 0),  # strictly after both
                    "stoptime": datetime(2150, 6, 3, 10, 0),
                    "dose_val_rx": 50.0, "dose_unit_rx": "mcg/min", "route": "IV",
                },
            ],
        },
    )


@pytest.fixture
def no_icu_extraction():
    """1 patient, 1 admission, 0 ICU stays, 1 diagnosis event only."""
    return ExtractionResult(
        patients=[{"subject_id": 7001, "gender": "F", "anchor_age": 45}],
        admissions=[{
            "hadm_id": 7101, "subject_id": 7001,
            "admittime": datetime(2150, 1, 1, 8, 0),
            "dischtime": datetime(2150, 1, 5, 14, 0),
            "admission_type": "EMERGENCY",
            "discharge_location": "HOME",
        }],
        icu_stays=[],
        events={
            "diagnosis": [{
                "hadm_id": 7101, "subject_id": 7001, "seq_num": 1,
                "icd_code": "J18.9", "icd_version": 10,
                "long_title": "Pneumonia, unspecified organism",
            }],
        },
    )


# ---------------------------------------------------------------------------
# TestBuildQueryGraphBasic
# ---------------------------------------------------------------------------


class TestBuildQueryGraphBasic:
    def test_empty_extraction_returns_ontology_only(self, ontology_dir, empty_extraction):
        graph, stats = build_query_graph(ontology_dir, empty_extraction)

        # Graph has ontology triples but nothing else
        assert len(graph) > 0
        # All stats should be zero
        for key, val in stats.items():
            assert val == 0, f"Expected stats['{key}'] == 0, got {val}"
        # No Patient nodes
        result = graph.query(
            "ASK { ?p rdf:type mimic:Patient }",
            initNs={"mimic": MIMIC_NS},
        )
        assert not bool(result)

    def test_patient_node_created(self, ontology_dir, minimal_extraction):
        graph, stats = build_query_graph(ontology_dir, minimal_extraction)

        assert stats["patients"] == 1

        # Patient type assertion
        result = graph.query(
            "ASK { mimic:PA-9001 rdf:type mimic:Patient }",
            initNs={"mimic": MIMIC_NS},
        )
        assert bool(result)

        # Demographics
        result = graph.query(
            """ASK {
                mimic:PA-9001 mimic:hasSubjectId ?id ;
                              mimic:hasGender ?g ;
                              mimic:hasAge ?a .
                FILTER(?id = 9001 && ?g = "M" && ?a = 60)
            }""",
            initNs={"mimic": MIMIC_NS},
        )
        assert bool(result)

    def test_admission_node_with_temporal_bounds(self, ontology_dir, minimal_extraction):
        graph, stats = build_query_graph(ontology_dir, minimal_extraction)

        assert stats["admissions"] == 1

        # Type assertion
        result = graph.query(
            "ASK { mimic:HA-9101 rdf:type mimic:HospitalAdmission }",
            initNs={"mimic": MIMIC_NS},
        )
        assert bool(result)

        # Temporal bounds
        result = graph.query(
            """ASK {
                mimic:HA-9101 time:hasBeginning ?begin ;
                              time:hasEnd ?end .
            }""",
            initNs={"mimic": MIMIC_NS, "time": TIME_NS},
        )
        assert bool(result)

        # Readmission defaults to false
        result = graph.query(
            """ASK {
                mimic:HA-9101 mimic:readmittedWithin30Days false .
            }""",
            initNs={"mimic": MIMIC_NS},
        )
        assert bool(result)

        # Bidirectional patient-admission link
        result = graph.query(
            """ASK {
                mimic:PA-9001 mimic:hasAdmission mimic:HA-9101 .
                mimic:HA-9101 mimic:admissionOf mimic:PA-9001 .
            }""",
            initNs={"mimic": MIMIC_NS},
        )
        assert bool(result)

    def test_icu_stay_and_days_created(self, ontology_dir, minimal_extraction):
        graph, stats = build_query_graph(ontology_dir, minimal_extraction)

        assert stats["icu_stays"] == 1
        assert stats["icu_days"] > 0

        # ICU stay linked to admission
        result = graph.query(
            """ASK {
                mimic:HA-9101 mimic:containsICUStay mimic:IS-9201 .
                mimic:IS-9201 rdf:type mimic:ICUStay .
            }""",
            initNs={"mimic": MIMIC_NS},
        )
        assert bool(result)

        # ICU stay has temporal bounds
        result = graph.query(
            """ASK {
                mimic:IS-9201 time:hasBeginning ?b ;
                              time:hasEnd ?e .
            }""",
            initNs={"mimic": MIMIC_NS, "time": TIME_NS},
        )
        assert bool(result)

        # Count ICU days (3-day stay = 4 calendar days: partial day 1, day 2, day 3, partial day 4)
        rows = list(graph.query(
            """SELECT (COUNT(?day) AS ?cnt) WHERE {
                ?day rdf:type mimic:ICUDay .
            }""",
            initNs={"mimic": MIMIC_NS},
        ))
        day_count = int(rows[0][0])
        assert day_count >= 3


# ---------------------------------------------------------------------------
# TestBuildQueryGraphEvents
# ---------------------------------------------------------------------------


class TestBuildQueryGraphEvents:
    def test_biomarker_event_with_resolved_stay(self, ontology_dir, minimal_extraction):
        graph, stats = build_query_graph(ontology_dir, minimal_extraction)

        assert stats["biomarkers"] == 1

        # BioMarkerEvent exists with properties and ICU stay link
        result = graph.query(
            """ASK {
                ?e rdf:type mimic:BioMarkerEvent ;
                   mimic:hasBiomarkerType ?label ;
                   mimic:hasValue ?val ;
                   mimic:associatedWithICUStay mimic:IS-9201 .
                FILTER(STR(?label) = "Creatinine")
            }""",
            initNs={"mimic": MIMIC_NS},
        )
        assert bool(result)

    def test_vital_event_created(self, ontology_dir, minimal_extraction):
        graph, stats = build_query_graph(ontology_dir, minimal_extraction)

        assert stats["vitals"] == 1

        result = graph.query(
            """ASK {
                ?e rdf:type mimic:ClinicalSignEvent ;
                   mimic:hasClinicalSignName ?name ;
                   mimic:associatedWithICUStay mimic:IS-9201 .
                FILTER(STR(?name) = "Heart Rate")
            }""",
            initNs={"mimic": MIMIC_NS},
        )
        assert bool(result)

    def test_diagnosis_linked_to_admission(self, ontology_dir, minimal_extraction):
        graph, stats = build_query_graph(ontology_dir, minimal_extraction)

        assert stats["diagnoses"] == 1

        # Diagnosis with ICD code and bidirectional links
        result = graph.query(
            """ASK {
                ?dx rdf:type mimic:DiagnosisEvent ;
                    mimic:hasIcdCode ?code ;
                    mimic:diagnosisOf mimic:HA-9101 .
                mimic:HA-9101 mimic:hasDiagnosis ?dx .
                FILTER(STR(?code) = "I63.0")
            }""",
            initNs={"mimic": MIMIC_NS},
        )
        assert bool(result)

    def test_prescription_event_created(self, ontology_dir, minimal_extraction):
        graph, stats = build_query_graph(ontology_dir, minimal_extraction)

        assert stats["prescriptions"] == 1

        result = graph.query(
            """ASK {
                ?rx rdf:type mimic:PrescriptionEvent ;
                    mimic:hasDrugName ?drug ;
                    time:hasBeginning ?begin ;
                    mimic:associatedWithICUStay mimic:IS-9201 .
                FILTER(STR(?drug) = "Vancomycin")
            }""",
            initNs={"mimic": MIMIC_NS, "time": TIME_NS},
        )
        assert bool(result)

    def test_microbiology_event_created(self, ontology_dir, minimal_extraction):
        graph, stats = build_query_graph(ontology_dir, minimal_extraction)

        assert stats["microbiology"] == 1

        result = graph.query(
            """ASK {
                ?mb rdf:type mimic:MicrobiologyEvent ;
                    mimic:hasSpecimenType ?spec ;
                    mimic:hasOrganismName ?org ;
                    mimic:associatedWithICUStay mimic:IS-9201 .
                FILTER(STR(?spec) = "BLOOD CULTURE" && STR(?org) = "STAPHYLOCOCCUS AUREUS")
            }""",
            initNs={"mimic": MIMIC_NS},
        )
        assert bool(result)


# ---------------------------------------------------------------------------
# Comorbidity derivation on the cohort path (plan I-B)
# ---------------------------------------------------------------------------


class TestComorbidityWiring:
    """I-B: the cohort-path graph derives Charlson comorbidities from each
    patient's ICD-10 diagnoses (prefix-matched against the Quan et al. 2005
    mapping) and emits one ``mimic:Comorbidity`` node per present category,
    linked to the patient. Before wiring, ``write_comorbidity`` was dead code.
    """

    def test_chf_icd_yields_comorbidity_node_linked_to_patient(self, ontology_dir):
        # A single patient whose only diagnosis is congestive heart failure
        # (ICD-10 I50.9, stored dotless as MIMIC does). Charlson should derive
        # exactly one comorbidity: congestive_heart_failure.
        extraction = ExtractionResult(
            patients=[{"subject_id": 6500, "gender": "F", "anchor_age": 74}],
            admissions=[{
                "hadm_id": 6510, "subject_id": 6500,
                "admittime": datetime(2150, 7, 1, 8, 0),
                "dischtime": datetime(2150, 7, 6, 12, 0),
                "admission_type": "EMERGENCY",
                "discharge_location": "HOME",
            }],
            events={
                "diagnosis": [{
                    "hadm_id": 6510, "subject_id": 6500, "seq_num": 1,
                    "icd_code": "I509", "icd_version": 10,
                    "long_title": "Heart failure, unspecified",
                }],
            },
        )

        graph, stats = build_query_graph(ontology_dir, extraction)

        assert stats["comorbidities"] == 1

        result = graph.query(
            """ASK {
                mimic:PA-6500 mimic:hasComorbidity ?c .
                ?c rdf:type mimic:Comorbidity ;
                   mimic:hasComorbidityName ?name .
                FILTER(STR(?name) = "congestive_heart_failure")
            }""",
            initNs={"mimic": MIMIC_NS},
        )
        assert bool(result)

    def test_no_charlson_diagnosis_yields_no_comorbidity(self, ontology_dir):
        # A diagnosis outside the Charlson mapping derives no comorbidity, and
        # absence is modeled as the lack of a node (no false-positive flags).
        extraction = ExtractionResult(
            patients=[{"subject_id": 6600, "gender": "M", "anchor_age": 40}],
            admissions=[{
                "hadm_id": 6610, "subject_id": 6600,
                "admittime": datetime(2150, 8, 1, 8, 0),
                "dischtime": datetime(2150, 8, 3, 12, 0),
                "admission_type": "ELECTIVE",
                "discharge_location": "HOME",
            }],
            events={
                "diagnosis": [{
                    "hadm_id": 6610, "subject_id": 6600, "seq_num": 1,
                    "icd_code": "Z9999", "icd_version": 10,
                    "long_title": "Unrelated code",
                }],
            },
        )

        graph, stats = build_query_graph(ontology_dir, extraction)

        assert stats["comorbidities"] == 0
        assert (None, RDF.type, MIMIC_NS.Comorbidity) not in graph


# ---------------------------------------------------------------------------
# Drug-category enrichment on the cohort path (plan I-A)
# ---------------------------------------------------------------------------


class TestDrugCategoryWiring:
    """I-A: when a resolver is injected, prescription events carry
    ``mimic:hasDrugCategory`` so the temporal feature extractor (III-A) can
    select a drug class (e.g. all vasopressors) and compute a dose trend.
    Default (no resolver) emits nothing, keeping unit builds offline.
    """

    def test_category_emitted_when_resolver_injected(
        self, ontology_dir, minimal_extraction,
    ):
        graph, _ = build_query_graph(
            ontology_dir, minimal_extraction,
            drug_category_resolver=DrugCategoryResolver(_MAPPINGS_DIR),
        )
        # minimal_extraction prescribes Vancomycin -> antibiotics.
        result = graph.query(
            """ASK {
                ?rx rdf:type mimic:PrescriptionEvent ;
                    mimic:hasDrugName ?d ;
                    mimic:hasDrugCategory ?c .
                FILTER(STR(?d) = "Vancomycin" && STR(?c) = "antibiotics")
            }""",
            initNs={"mimic": MIMIC_NS},
        )
        assert bool(result)

    def test_no_category_without_resolver(self, ontology_dir, minimal_extraction):
        graph, _ = build_query_graph(ontology_dir, minimal_extraction)
        assert (None, MIMIC_NS.hasDrugCategory, None) not in graph


# ---------------------------------------------------------------------------
# TestBuildQueryGraphAdvanced
# ---------------------------------------------------------------------------


class TestBuildQueryGraphAdvanced:
    def test_allen_relations_computed(self, ontology_dir, minimal_extraction):
        graph, stats = build_query_graph(ontology_dir, minimal_extraction)

        # With multiple temporal events, Allen relations should be computed
        assert stats["allen_relations"] > 0

        # At least one Allen relation triple exists
        result = graph.query(
            """ASK {
                { ?a time:before ?b }
                UNION
                { ?a time:inside ?b }
                UNION
                { ?a time:intervalOverlaps ?b }
                UNION
                { ?a time:intervalMeets ?b }
            }""",
            initNs={"time": TIME_NS},
        )
        assert bool(result)

    def test_multiple_patients(self, ontology_dir, multi_patient_extraction):
        graph, stats = build_query_graph(ontology_dir, multi_patient_extraction)

        assert stats["patients"] == 2

        rows = list(graph.query(
            """SELECT (COUNT(?p) AS ?cnt) WHERE {
                ?p rdf:type mimic:Patient .
            }""",
            initNs={"mimic": MIMIC_NS},
        ))
        assert int(rows[0][0]) == 2

    def test_sequential_admissions_linked(self, ontology_dir, multi_patient_extraction):
        graph, stats = build_query_graph(ontology_dir, multi_patient_extraction)

        # Patient 8001 has admissions 8101 then 8102 — should be linked
        result = graph.query(
            """ASK {
                mimic:HA-8101 mimic:followedBy mimic:HA-8102 .
            }""",
            initNs={"mimic": MIMIC_NS},
        )
        assert bool(result)

    def test_no_icu_diagnoses_still_created(self, ontology_dir, no_icu_extraction):
        graph, stats = build_query_graph(ontology_dir, no_icu_extraction)

        assert stats["icu_stays"] == 0
        assert stats["diagnoses"] == 1

        # Diagnosis exists and is linked to admission
        result = graph.query(
            """ASK {
                ?dx rdf:type mimic:DiagnosisEvent ;
                    mimic:hasIcdCode ?code ;
                    mimic:diagnosisOf mimic:HA-7101 .
                FILTER(STR(?code) = "J18.9")
            }""",
            initNs={"mimic": MIMIC_NS},
        )
        assert bool(result)

    def test_stats_dict_complete(self, ontology_dir, minimal_extraction):
        graph, stats = build_query_graph(ontology_dir, minimal_extraction)

        expected_keys = {
            "patients", "admissions", "icu_stays", "icu_days",
            "biomarkers", "vitals", "prescriptions", "diagnoses",
            "microbiology", "comorbidities", "allen_relations",
        }
        assert set(stats.keys()) == expected_keys

        # Verify counts for minimal extraction (1 of each)
        assert stats["patients"] == 1
        assert stats["admissions"] == 1
        assert stats["icu_stays"] == 1
        assert stats["icu_days"] >= 3
        assert stats["biomarkers"] == 1
        assert stats["vitals"] == 1
        assert stats["prescriptions"] == 1
        assert stats["diagnoses"] == 1
        assert stats["microbiology"] == 1
        # The I63.0 (cerebral infarction) diagnosis is a Charlson
        # cerebrovascular_disease, so one comorbidity is derived (plan I-B).
        assert stats["comorbidities"] == 1
        assert stats["allen_relations"] > 0


# ---------------------------------------------------------------------------
# Conditional Allen relations
# ---------------------------------------------------------------------------


class TestSkipAllenRelations:
    def test_skip_allen_no_temporal_relations(self, ontology_dir, minimal_extraction):
        """When skip_allen_relations=True, no Allen relations are computed."""
        graph, stats = build_query_graph(
            ontology_dir, minimal_extraction, skip_allen_relations=True,
        )
        assert stats["allen_relations"] == 0

    def test_allen_computed_by_default(self, ontology_dir, minimal_extraction):
        """Default behavior still computes Allen relations."""
        graph, stats = build_query_graph(ontology_dir, minimal_extraction)
        assert stats["allen_relations"] > 0


# ---------------------------------------------------------------------------
# Allen precedence edges on the cohort path (plan I-D)
# ---------------------------------------------------------------------------


class TestAllenPrecedenceEdges:
    """I-D: verify the cohort-path graph carries the actual Allen *edges* the
    temporal feature extractor (plan III-A) will read — not just a nonzero
    count. ``build_query_graph`` computes Allen by default
    (``skip_allen_relations=False``), which is exactly what the orchestrator's
    temporal cohort path requests (``skip_allen_relations=not any_temporal``).
    Adjacency uses ``time:intervalMeets`` (OWL-Time's interval-meets); coarse
    precedence uses ``time:before`` — which is fanned-out, so it is only safe
    for "A precedes B", never for adjacency.
    """

    def test_meets_and_before_edges_between_events(
        self, ontology_dir, temporal_cohort_extraction,
    ):
        graph, stats = build_query_graph(ontology_dir, temporal_cohort_extraction)
        assert stats["allen_relations"] > 0

        # Adjacency: the first administration meets the one that begins as it
        # ends. Both endpoints must be clinical events (not the begin/end
        # Instant nodes, which are not stay-associated and so never paired).
        meets = list(graph.subject_objects(TIME_NS.intervalMeets))
        assert meets, "expected a time:intervalMeets adjacency edge"
        for s, o in meets:
            assert (s, RDF.type, MIMIC_NS.PrescriptionEvent) in graph
            assert (o, RDF.type, MIMIC_NS.PrescriptionEvent) in graph

        # Coarse precedence: the strictly-later administration is before-linked.
        before = list(graph.subject_objects(TIME_NS.before))
        assert before, "expected a time:before precedence edge"
        for s, o in before:
            assert (s, RDF.type, MIMIC_NS.PrescriptionEvent) in graph
            assert (o, RDF.type, MIMIC_NS.PrescriptionEvent) in graph

    def test_precedence_edges_absent_when_allen_skipped(
        self, ontology_dir, temporal_cohort_extraction,
    ):
        """Built with Allen skipped, the same graph carries no precedence
        edges — a pure-contextual cohort neither pays for nor depends on them.
        """
        graph, _ = build_query_graph(
            ontology_dir, temporal_cohort_extraction, skip_allen_relations=True,
        )
        assert (None, TIME_NS.intervalMeets, None) not in graph
        assert (None, TIME_NS.before, None) not in graph


# ---------------------------------------------------------------------------
# Parallel graph build
# ---------------------------------------------------------------------------


class TestParallelGraphBuild:
    def test_parallel_produces_same_triple_count(self, ontology_dir, minimal_extraction):
        """Parallel build produces the same number of triples as serial."""
        graph_serial, stats_serial = build_query_graph(
            ontology_dir, minimal_extraction, max_workers=1,
        )
        graph_parallel, stats_parallel = build_query_graph(
            ontology_dir, minimal_extraction, max_workers=2,
        )
        assert len(graph_serial) == len(graph_parallel)
        for key in ("patients", "admissions", "icu_stays", "biomarkers", "vitals"):
            assert stats_serial[key] == stats_parallel[key]

    def test_max_workers_1_is_serial(self, ontology_dir, minimal_extraction):
        """max_workers=1 uses serial path (baseline behavior)."""
        graph, stats = build_query_graph(
            ontology_dir, minimal_extraction, max_workers=1,
        )
        assert stats["patients"] == 1
        assert len(graph) > 0


class _StayVisitCounter(list):
    """A ``list`` that counts every element yielded across all iterations.

    Lets the complexity test below prove ``build_query_graph`` passes over the
    ICU stays a constant number of times rather than once per patient.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.visits = 0

    def __iter__(self):
        for item in super().__iter__():
            self.visits += 1
            yield item


class TestBuildQueryGraphComplexity:
    def test_stays_not_rescanned_per_patient(self, ontology_dir):
        """Build is O(patients + stays), not O(patients x stays).

        Regression for a cohort-wide hang: the serial path scanned every ICU
        stay once per patient, so on the full cohort (~50k patients x ~80k
        stays) it spun on billions of iterations and never returned. Each
        patient here owns one matching stay; an instrumented ``icu_stays``
        counts total visits. The buggy path visits ~patients * stays; the
        fixed path visits ~stays (a single pass to build the hadm->stay index).
        """
        p = 200
        patients, admissions, stays = [], [], []
        for i in range(p):
            sid, hadm, stay = 100_000 + i, 200_000 + i, 300_000 + i
            patients.append({"subject_id": sid, "gender": "M", "anchor_age": 60})
            admissions.append({
                "hadm_id": hadm, "subject_id": sid,
                "admittime": datetime(2150, 6, 1, 8, 0),
                "dischtime": datetime(2150, 6, 10, 14, 0),
                "admission_type": "EMERGENCY", "discharge_location": "HOME",
            })
            stays.append({
                "stay_id": stay, "hadm_id": hadm, "subject_id": sid,
                "intime": datetime(2150, 6, 1, 10, 0),
                "outtime": datetime(2150, 6, 4, 10, 0), "los": 3.0,
            })

        ext = ExtractionResult(
            patients=patients, admissions=admissions, icu_stays=stays, events={},
        )
        # validate_assignment is off on ExtractionResult, so the subclass
        # survives assignment and instruments the real build.
        counter = _StayVisitCounter(ext.icu_stays)
        ext.icu_stays = counter

        graph, stats = build_query_graph(
            ontology_dir, ext, skip_allen_relations=True, max_workers=1,
        )

        # Correctness preserved: every stay is still attached to its admission.
        assert stats["icu_stays"] == p
        # Complexity guard: total stay visits stay linear in the stay count.
        # Buggy path ~= p * len(stays) (40,000); fixed path ~= len(stays).
        assert counter.visits <= 3 * len(stays)
