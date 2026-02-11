"""Tests for src.gnn.graph_export — RDF→PyG HeteroData conversion."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from src.gnn.graph_export import (
    EMBEDDING_DIM,
    GENDER_ENCODING,
    _build_structural_node_maps,
    _validate_heterodata,
    export_rdf_to_heterodata,
)
from src.graph_construction.event_writers import (
    write_biomarker_event,
    write_clinical_sign_event,
    write_diagnosis_event,
    write_icu_days,
    write_icu_stay,
    write_microbiology_event,
    write_prescription_event,
)
from src.graph_construction.ontology import MIMIC_NS, SNOMED_NS, initialize_graph
from src.graph_construction.patient_writer import (
    link_sequential_admissions,
    write_admission,
    write_patient,
)

ONTOLOGY_DIR = Path(__file__).parent.parent.parent / "ontology" / "definition"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _add_snomed(g: Graph, event_uri: URIRef, code: str) -> None:
    """Manually add SNOMED triples to an event."""
    g.add((event_uri, MIMIC_NS.hasSnomedCode, Literal(code, datatype=XSD.string)))
    g.add((event_uri, MIMIC_NS.hasSnomedConcept, SNOMED_NS[code]))


def _mock_embed_unmapped(terms):
    """Return random embeddings keyed by term string."""
    return {t: torch.randn(EMBEDDING_DIM) for t in terms}


def _mock_split_fn(df, target_col):
    """Deterministic split: patient 10→train, 20→val, 30→test."""
    train = df[df["subject_id"] == 10]
    val = df[df["subject_id"] == 20]
    test = df[df["subject_id"] == 30]
    return train, val, test


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def export_test_graph() -> Graph:
    """Rich RDF graph for graph_export tests.

    3 patients, 4 admissions, 3 ICU stays, ICU days, plus clinical events
    with manual SNOMED triples.
    """
    g = initialize_graph(ONTOLOGY_DIR)

    # ── Patients ──
    p10 = write_patient(g, {"subject_id": 10, "gender": "M", "anchor_age": 65})
    p20 = write_patient(g, {"subject_id": 20, "gender": "F", "anchor_age": 55})
    p30 = write_patient(g, {"subject_id": 30, "gender": "M", "anchor_age": 70})

    # ── Admissions ──
    adm100 = write_admission(g, {
        "hadm_id": 100, "subject_id": 10,
        "admittime": datetime(2150, 1, 1, 8, 0, 0),
        "dischtime": datetime(2150, 1, 10, 14, 0, 0),
        "admission_type": "EMERGENCY", "discharge_location": "HOME",
        "readmitted_30d": True, "readmitted_60d": True,
    }, p10)

    adm200 = write_admission(g, {
        "hadm_id": 200, "subject_id": 10,
        "admittime": datetime(2150, 1, 16, 10, 0, 0),
        "dischtime": datetime(2150, 1, 20, 12, 0, 0),
        "admission_type": "URGENT", "discharge_location": "SNF",
        "readmitted_30d": False, "readmitted_60d": False,
    }, p10)

    adm300 = write_admission(g, {
        "hadm_id": 300, "subject_id": 20,
        "admittime": datetime(2150, 2, 1, 6, 0, 0),
        "dischtime": datetime(2150, 2, 8, 16, 0, 0),
        "admission_type": "ELECTIVE", "discharge_location": "HOME",
        "readmitted_30d": False, "readmitted_60d": False,
    }, p20)

    adm400 = write_admission(g, {
        "hadm_id": 400, "subject_id": 30,
        "admittime": datetime(2150, 3, 1, 14, 0, 0),
        "dischtime": datetime(2150, 3, 5, 10, 0, 0),
        "admission_type": "EMERGENCY", "discharge_location": "HOME",
        "readmitted_30d": False, "readmitted_60d": False,
    }, p30)

    # followedBy: adm100 → adm200  (gap ≈ 15 days)
    link_sequential_admissions(g, [adm100, adm200])

    # ── ICU stays ──
    stay1000 = write_icu_stay(g, {
        "stay_id": 1000, "hadm_id": 100, "subject_id": 10,
        "intime": datetime(2150, 1, 1, 10, 0, 0),
        "outtime": datetime(2150, 1, 4, 10, 0, 0), "los": 3.0,
    }, adm100)
    days1000 = write_icu_days(g, {
        "stay_id": 1000,
        "intime": datetime(2150, 1, 1, 10, 0, 0),
        "outtime": datetime(2150, 1, 4, 10, 0, 0),
    }, stay1000)

    stay2000 = write_icu_stay(g, {
        "stay_id": 2000, "hadm_id": 300, "subject_id": 20,
        "intime": datetime(2150, 2, 1, 8, 0, 0),
        "outtime": datetime(2150, 2, 3, 8, 0, 0), "los": 2.0,
    }, adm300)
    days2000 = write_icu_days(g, {
        "stay_id": 2000,
        "intime": datetime(2150, 2, 1, 8, 0, 0),
        "outtime": datetime(2150, 2, 3, 8, 0, 0),
    }, stay2000)

    stay3000 = write_icu_stay(g, {
        "stay_id": 3000, "hadm_id": 400, "subject_id": 30,
        "intime": datetime(2150, 3, 1, 16, 0, 0),
        "outtime": datetime(2150, 3, 4, 16, 0, 0), "los": 3.0,
    }, adm400)
    days3000 = write_icu_days(g, {
        "stay_id": 3000,
        "intime": datetime(2150, 3, 1, 16, 0, 0),
        "outtime": datetime(2150, 3, 4, 16, 0, 0),
    }, stay3000)

    # ── BioMarkerEvents (3 events → 2 lab concepts via SNOMED dedup) ──
    bme1 = write_biomarker_event(g, {
        "labevent_id": 5001, "stay_id": 1000, "itemid": 50912,
        "charttime": datetime(2150, 1, 1, 12, 0, 0),
        "label": "Creatinine", "fluid": "Blood", "category": "Chemistry",
        "valuenum": 1.2, "valueuom": "mg/dL",
        "ref_range_lower": 0.7, "ref_range_upper": 1.3,
    }, stay1000, days1000)
    _add_snomed(g, bme1, "12345")

    bme2 = write_biomarker_event(g, {
        "labevent_id": 5002, "stay_id": 2000, "itemid": 50912,
        "charttime": datetime(2150, 2, 1, 10, 0, 0),
        "label": "Creatinine", "fluid": "Blood", "category": "Chemistry",
        "valuenum": 0.9, "valueuom": "mg/dL",
        "ref_range_lower": 0.7, "ref_range_upper": 1.3,
    }, stay2000, days2000)
    _add_snomed(g, bme2, "12345")

    bme3 = write_biomarker_event(g, {
        "labevent_id": 5003, "stay_id": 3000, "itemid": 50971,
        "charttime": datetime(2150, 3, 1, 18, 0, 0),
        "label": "Sodium", "fluid": "Blood", "category": "Chemistry",
        "valuenum": 140.0, "valueuom": "mEq/L",
        "ref_range_lower": 136.0, "ref_range_upper": 145.0,
    }, stay3000, days3000)
    _add_snomed(g, bme3, "67890")

    # ── ClinicalSignEvents (2 events → 1 vital concept) ──
    cse1 = write_clinical_sign_event(g, {
        "stay_id": 1000, "itemid": 220045,
        "charttime": datetime(2150, 1, 2, 8, 0, 0),
        "label": "Heart Rate", "category": "Routine Vital Signs",
        "valuenum": 78.0,
    }, stay1000, days1000)
    _add_snomed(g, cse1, "11111")

    cse2 = write_clinical_sign_event(g, {
        "stay_id": 2000, "itemid": 220045,
        "charttime": datetime(2150, 2, 2, 6, 0, 0),
        "label": "Heart Rate", "category": "Routine Vital Signs",
        "valuenum": 82.0,
    }, stay2000, days2000)
    _add_snomed(g, cse2, "11111")

    # ── PrescriptionEvents (2 events → 2 drug concepts) ──
    rxe1 = write_prescription_event(g, {
        "hadm_id": 100, "stay_id": 1000, "drug": "Vancomycin",
        "starttime": datetime(2150, 1, 1, 14, 0, 0),
        "stoptime": datetime(2150, 1, 3, 14, 0, 0),
        "dose_val_rx": 1000.0, "dose_unit_rx": "mg", "route": "IV",
    }, stay1000, days1000)
    _add_snomed(g, rxe1, "22222")

    rxe2 = write_prescription_event(g, {
        "hadm_id": 400, "stay_id": 3000, "drug": "Aspirin",
        "starttime": datetime(2150, 3, 2, 10, 0, 0),
        "stoptime": datetime(2150, 3, 4, 10, 0, 0),
        "dose_val_rx": 81.0, "dose_unit_rx": "mg", "route": "PO",
    }, stay3000, days3000)
    # Aspirin: unmapped (no SNOMED)

    # ── MicrobiologyEvent (1 event → 1 micro concept) ──
    mbe1 = write_microbiology_event(g, {
        "microevent_id": 7001, "stay_id": 1000,
        "charttime": datetime(2150, 1, 2, 12, 0, 0),
        "spec_type_desc": "BLOOD CULTURE", "org_name": "Staph aureus",
    }, stay1000, days1000)
    _add_snomed(g, mbe1, "33333")

    # ── DiagnosisEvents (2 events → 2 diagnosis concepts) ──
    dxe1 = write_diagnosis_event(g, {
        "hadm_id": 100, "seq_num": 1, "icd_code": "I63.0", "icd_version": 10,
        "long_title": "Cerebral infarction due to thrombosis",
    }, adm100)
    _add_snomed(g, dxe1, "44444")

    dxe2 = write_diagnosis_event(g, {
        "hadm_id": 400, "seq_num": 1, "icd_code": "G40.9", "icd_version": 10,
        "long_title": "Epilepsy unspecified",
    }, adm400)
    # G40.9: unmapped (no SNOMED)

    return g


@pytest.fixture
def mock_feature_matrix(tmp_path: Path) -> Path:
    """Parquet with 4 admission rows, 5 features."""
    df = pd.DataFrame({
        "hadm_id": [100, 200, 300, 400],
        "subject_id": [10, 10, 20, 30],
        "readmitted_30d": [1, 0, 0, 0],
        "readmitted_60d": [1, 0, 0, 0],
        "feat_1": [1.0, 2.0, 3.0, 4.0],
        "feat_2": [5.0, 6.0, 7.0, 8.0],
        "feat_3": [0.1, 0.2, 0.3, 0.4],
        "feat_4": [10.0, 20.0, 30.0, 40.0],
        "feat_5": [0.5, 0.6, 0.7, 0.8],
    })
    path = tmp_path / "feature_matrix.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def mock_concept_embeddings(tmp_path: Path) -> Path:
    """Pre-computed SNOMED embeddings for test codes."""
    embs = {
        "12345": torch.randn(EMBEDDING_DIM),
        "67890": torch.randn(EMBEDDING_DIM),
        "11111": torch.randn(EMBEDDING_DIM),
        "22222": torch.randn(EMBEDDING_DIM),
        "33333": torch.randn(EMBEDDING_DIM),
        "44444": torch.randn(EMBEDDING_DIM),
    }
    path = tmp_path / "concept_embeddings.pt"
    torch.save(embs, path)
    return path


@pytest.fixture
def exported_data(
    export_test_graph, mock_feature_matrix, mock_concept_embeddings, tmp_path,
):
    """Run the full export pipeline and return the HeteroData."""
    output_path = tmp_path / "test_graph.pt"
    data = export_rdf_to_heterodata(
        rdf_graph=export_test_graph,
        feature_matrix_path=mock_feature_matrix,
        concept_embeddings_path=mock_concept_embeddings,
        split_fn=_mock_split_fn,
        output_path=output_path,
        embed_unmapped_fn=_mock_embed_unmapped,
    )
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestNodeIdMapping:
    def test_structural_node_counts(self, exported_data):
        """3 patients, 4 admissions, 3 ICU stays."""
        assert exported_data["patient"].num_nodes == 3
        assert exported_data["admission"].num_nodes == 4
        assert exported_data["icu_stay"].num_nodes == 3

    def test_clinical_dedup_lab(self, exported_data):
        """3 BioMarkerEvents should deduplicate to 2 lab concept nodes."""
        assert exported_data["lab"].num_nodes == 2

    def test_clinical_dedup_vital(self, exported_data):
        """2 ClinicalSignEvents with same SNOMED → 1 vital concept node."""
        assert exported_data["vital"].num_nodes == 1

    def test_drug_nodes(self, exported_data):
        """2 PrescriptionEvents → 2 drug concept nodes (one SNOMED, one unmapped)."""
        assert exported_data["drug"].num_nodes == 2

    def test_micro_nodes(self, exported_data):
        assert exported_data["microbiology"].num_nodes == 1

    def test_diagnosis_nodes(self, exported_data):
        """2 DiagnosisEvents → 2 diagnosis concept nodes."""
        assert exported_data["diagnosis"].num_nodes == 2

    def test_contiguous_ids(self, export_test_graph):
        """Structural node maps produce contiguous IDs starting from 0."""
        maps = _build_structural_node_maps(export_test_graph)
        for ntype, m in maps.items():
            ids = sorted(m.values())
            assert ids == list(range(len(ids))), f"{ntype} IDs not contiguous: {ids}"


class TestEdgeShapes:
    def test_edge_index_shape(self, exported_data):
        """All edge_index tensors have shape (2, E) with E > 0."""
        for src, rel, dst in exported_data.edge_types:
            ei = exported_data[src, rel, dst].edge_index
            assert ei.dim() == 2
            assert ei.shape[0] == 2

    def test_edge_index_bounds(self, exported_data):
        """All edge indices are within [0, num_nodes)."""
        for src, rel, dst in exported_data.edge_types:
            ei = exported_data[src, rel, dst].edge_index
            if ei.numel() > 0:
                assert ei[0].max() < exported_data[src].num_nodes
                assert ei[1].max() < exported_data[dst].num_nodes


class TestReverseEdges:
    def test_reverse_edge_exists(self, exported_data):
        """Every forward edge type has a rev_* counterpart."""
        forward_types = [
            (s, r, d) for s, r, d in exported_data.edge_types if not r.startswith("rev_")
        ]
        for src, rel, dst in forward_types:
            rev_rel = f"rev_{rel}"
            assert (dst, rev_rel, src) in exported_data.edge_types, \
                f"Missing reverse edge for ({src},{rel},{dst})"

    def test_reverse_edge_count(self, exported_data):
        """Reverse edges have the same number of edges as forward."""
        forward_types = [
            (s, r, d) for s, r, d in exported_data.edge_types if not r.startswith("rev_")
        ]
        for src, rel, dst in forward_types:
            fwd_e = exported_data[src, rel, dst].edge_index.shape[1]
            rev_e = exported_data[dst, f"rev_{rel}", src].edge_index.shape[1]
            assert fwd_e == rev_e

    def test_reverse_edge_swapped(self, exported_data):
        """Reverse edges have swapped src/dst compared to forward."""
        for src, rel, dst in exported_data.edge_types:
            if rel.startswith("rev_"):
                continue
            fwd_ei = exported_data[src, rel, dst].edge_index
            rev_ei = exported_data[dst, f"rev_{rel}", src].edge_index
            assert torch.equal(fwd_ei[0], rev_ei[1])
            assert torch.equal(fwd_ei[1], rev_ei[0])


class TestTemporalEdgeAttr:
    def test_event_edges_have_attr(self, exported_data):
        """icu_day→has_event→* edges should have edge_attr shape (E, 1)."""
        for src, rel, dst in exported_data.edge_types:
            if src == "icu_day" and rel == "has_event":
                attr = exported_data[src, rel, dst].edge_attr
                assert attr is not None
                assert attr.shape[1] == 1

    def test_temporal_attr_non_negative(self, exported_data):
        """Temporal Δt should be ≥ 0."""
        for src, rel, dst in exported_data.edge_types:
            if src == "icu_day" and rel == "has_event":
                attr = exported_data[src, rel, dst].edge_attr
                assert (attr >= 0).all()


class TestAdmissionFeatures:
    def test_shape(self, exported_data):
        """4 admissions × 5 features."""
        assert exported_data["admission"].x.shape == (4, 5)

    def test_values_match_parquet(self, exported_data):
        """Verify specific feature values land at correct admission positions.

        Columns are sorted alphabetically: feat_1..feat_5.
        The fixture maps hadm→features as:
            100 → [1.0, 5.0, 0.1, 10.0, 0.5]
            200 → [2.0, 6.0, 0.2, 20.0, 0.6]
            300 → [3.0, 7.0, 0.3, 30.0, 0.7]
            400 → [4.0, 8.0, 0.4, 40.0, 0.8]
        Node ordering is by sorted URI so HA-100 < HA-200 < HA-300 < HA-400.
        """
        x = exported_data["admission"].x
        expected = torch.tensor([
            [1.0, 5.0, 0.1, 10.0, 0.5],
            [2.0, 6.0, 0.2, 20.0, 0.6],
            [3.0, 7.0, 0.3, 30.0, 0.7],
            [4.0, 8.0, 0.4, 40.0, 0.8],
        ])
        assert torch.allclose(x, expected, atol=1e-5)


class TestLabels:
    def test_label_shape(self, exported_data):
        assert exported_data["admission"].y.shape == (4,)

    def test_label_values(self, exported_data):
        """Exactly 1 positive label (hadm 100)."""
        assert exported_data["admission"].y.sum() == 1.0

    def test_60d_labels(self, exported_data):
        assert exported_data["admission"].y_60d.shape == (4,)
        assert exported_data["admission"].y_60d.sum() == 1.0


class TestMasks:
    def test_disjointness(self, exported_data):
        tm = exported_data["admission"].train_mask
        vm = exported_data["admission"].val_mask
        tsm = exported_data["admission"].test_mask
        assert (tm & vm).sum() == 0
        assert (tm & tsm).sum() == 0
        assert (vm & tsm).sum() == 0

    def test_coverage(self, exported_data):
        tm = exported_data["admission"].train_mask
        vm = exported_data["admission"].val_mask
        tsm = exported_data["admission"].test_mask
        assert (tm | vm | tsm).all()

    def test_split_sizes(self, exported_data):
        """Patient 10 has 2 admissions (train), 20 has 1 (val), 30 has 1 (test)."""
        assert exported_data["admission"].train_mask.sum() == 2
        assert exported_data["admission"].val_mask.sum() == 1
        assert exported_data["admission"].test_mask.sum() == 1


class TestFullRoundtrip:
    def test_save_and_reload(
        self, export_test_graph, mock_feature_matrix, mock_concept_embeddings, tmp_path,
    ):
        output_path = tmp_path / "roundtrip_graph.pt"
        original = export_rdf_to_heterodata(
            rdf_graph=export_test_graph,
            feature_matrix_path=mock_feature_matrix,
            concept_embeddings_path=mock_concept_embeddings,
            output_path=output_path,
            embed_unmapped_fn=_mock_embed_unmapped,
        )
        loaded = torch.load(output_path, weights_only=False)

        # Same node types
        assert set(original.node_types) == set(loaded.node_types)

        # Same edge types
        assert set(original.edge_types) == set(loaded.edge_types)

        # Same node counts
        for ntype in original.node_types:
            assert original[ntype].num_nodes == loaded[ntype].num_nodes

        # Same edge counts
        for s, r, d in original.edge_types:
            assert original[s, r, d].edge_index.shape == loaded[s, r, d].edge_index.shape


class TestMetadataJson:
    def test_metadata_file_exists(
        self, export_test_graph, mock_feature_matrix, mock_concept_embeddings, tmp_path,
    ):
        output_path = tmp_path / "meta_test_graph.pt"
        export_rdf_to_heterodata(
            rdf_graph=export_test_graph,
            feature_matrix_path=mock_feature_matrix,
            concept_embeddings_path=mock_concept_embeddings,
            output_path=output_path,
            embed_unmapped_fn=_mock_embed_unmapped,
        )
        import json
        meta_path = output_path.with_suffix(".meta.json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "node_counts" in meta
        assert "edge_counts" in meta
        assert "label_distribution" in meta
        assert "timestamp" in meta


class TestPatientFeatures:
    def test_shape(self, exported_data):
        assert exported_data["patient"].x.shape == (3, 2)

    def test_values(self, exported_data):
        """Check ages and gender encoding are present."""
        x = exported_data["patient"].x
        ages = x[:, 0].tolist()
        genders = x[:, 1].tolist()
        # Our patients are ages 65, 55, 70 (sorted by URI)
        assert sorted(ages) == [55.0, 65.0, 70.0]
        # Genders: M=1.0, F=0.0; we have 2M, 1F
        assert sorted(genders) == [0.0, 1.0, 1.0]


class TestClinicalFeaturesDim:
    def test_lab_features_773(self, exported_data):
        """Lab nodes: 768 embedding + 5 stats = 773."""
        assert exported_data["lab"].x.shape == (2, 773)

    def test_vital_features_773(self, exported_data):
        """Vital nodes: 768 embedding + 5 stats = 773."""
        assert exported_data["vital"].x.shape == (1, 773)

    def test_drug_features_768(self, exported_data):
        """Drug nodes: 768 embedding only."""
        assert exported_data["drug"].x.shape == (2, 768)

    def test_diagnosis_features_768(self, exported_data):
        assert exported_data["diagnosis"].x.shape == (2, 768)

    def test_micro_features_768(self, exported_data):
        assert exported_data["microbiology"].x.shape == (1, 768)


class TestFollowedByDeltaT:
    def test_followed_by_edge_count(self, exported_data):
        """Exactly 1 followed_by edge (adm100→adm200)."""
        ei = exported_data["admission", "followed_by", "admission"].edge_index
        assert ei.shape[1] == 1

    def test_followed_by_gap(self, exported_data):
        """Gap from adm100 (Jan 1) to adm200 (Jan 16) ≈ 15 days."""
        attr = exported_data["admission", "followed_by", "admission"].edge_attr
        gap_days = attr.item()
        assert 14.5 < gap_days < 16.0


class TestNoSplitFnAllTrain:
    def test_all_train(
        self, export_test_graph, mock_feature_matrix, mock_concept_embeddings, tmp_path,
    ):
        output_path = tmp_path / "no_split_graph.pt"
        data = export_rdf_to_heterodata(
            rdf_graph=export_test_graph,
            feature_matrix_path=mock_feature_matrix,
            concept_embeddings_path=mock_concept_embeddings,
            split_fn=None,
            output_path=output_path,
            embed_unmapped_fn=_mock_embed_unmapped,
        )
        assert data["admission"].train_mask.all()
        assert data["admission"].val_mask.sum() == 0
        assert data["admission"].test_mask.sum() == 0


class TestIcuStayFeatures:
    def test_shape(self, exported_data):
        assert exported_data["icu_stay"].x.shape == (3, 2)

    def test_duration_hours(self, exported_data):
        """ICU stay durations: 3.0, 2.0, 3.0 days → 72.0, 48.0, 72.0 hours."""
        x = exported_data["icu_stay"].x
        hours = sorted(x[:, 0].tolist())
        assert hours == pytest.approx([48.0, 72.0, 72.0])

    def test_icu_day_counts(self, exported_data):
        """Each stay should have the correct number of ICU days."""
        x = exported_data["icu_stay"].x
        day_counts = sorted(x[:, 1].tolist())
        # stay1000: Jan 1 10:00 → Jan 4 10:00 = 4 calendar days
        # stay2000: Feb 1 08:00 → Feb 3 08:00 = 3 calendar days
        # stay3000: Mar 1 16:00 → Mar 4 16:00 = 4 calendar days
        assert all(c > 0 for c in day_counts)


class TestIcuDayFeatures:
    def test_shape(self, exported_data):
        x = exported_data["icu_day"].x
        assert x.shape[1] == 2
        assert x.shape[0] > 0

    def test_day_numbers_start_at_one(self, exported_data):
        """All day numbers should be >= 1."""
        x = exported_data["icu_day"].x
        assert (x[:, 0] >= 1).all()

    def test_event_counts_non_negative(self, exported_data):
        x = exported_data["icu_day"].x
        assert (x[:, 1] >= 0).all()


class TestLabPopulationStats:
    def test_creatinine_stats(self, exported_data):
        """Creatinine (SNOMED 12345) has 2 events: 1.2 and 0.9.

        Expected stats: mean=1.05, std≈0.212, min=0.9, max=1.2, abnormal=0.0
        (both values within ref range 0.7–1.3).
        """
        x = exported_data["lab"].x
        # SNOMED 12345 sorts before 67890 → index 0
        stats = x[0, EMBEDDING_DIM:]
        assert stats[0].item() == pytest.approx(1.05, abs=0.01)  # mean
        assert stats[1].item() == pytest.approx(0.2121, abs=0.02)  # std
        assert stats[2].item() == pytest.approx(0.9, abs=0.01)  # min
        assert stats[3].item() == pytest.approx(1.2, abs=0.01)  # max
        assert stats[4].item() == pytest.approx(0.0, abs=0.01)  # abnormal_rate

    def test_sodium_stats(self, exported_data):
        """Sodium (SNOMED 67890) has 1 event: 140.0.

        Expected: mean=140, std=0, min=140, max=140, abnormal=0.
        """
        x = exported_data["lab"].x
        # SNOMED 67890 sorts after 12345 → index 1
        stats = x[1, EMBEDDING_DIM:]
        assert stats[0].item() == pytest.approx(140.0, abs=0.1)
        assert stats[1].item() == pytest.approx(0.0, abs=0.01)  # single value → 0
        assert stats[4].item() == pytest.approx(0.0, abs=0.01)


class TestValidation:
    def test_nan_raises(self, exported_data):
        """Injecting NaN into features should raise ValueError."""
        exported_data["patient"].x[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            _validate_heterodata(exported_data)

    def test_edge_oob_raises(self, exported_data):
        """Edge index out of bounds should raise ValueError."""
        # Corrupt an edge to point beyond num_nodes
        src, rel, dst = exported_data.edge_types[0]
        n = exported_data[src].num_nodes
        exported_data[src, rel, dst].edge_index[0, 0] = n + 100
        with pytest.raises(ValueError, match="out of bounds"):
            _validate_heterodata(exported_data)

    def test_mask_overlap_raises(self, exported_data):
        """Overlapping masks should raise ValueError."""
        exported_data["admission"].train_mask[:] = True
        exported_data["admission"].val_mask[0] = True
        with pytest.raises(ValueError, match="not disjoint"):
            _validate_heterodata(exported_data)

    def test_mask_incomplete_raises(self, exported_data):
        """Masks not covering all nodes should raise ValueError."""
        exported_data["admission"].train_mask[:] = False
        exported_data["admission"].val_mask[:] = False
        exported_data["admission"].test_mask[:] = False
        with pytest.raises(ValueError, match="do not cover"):
            _validate_heterodata(exported_data)


class TestMappingsJson:
    def test_mappings_file_contents(
        self, export_test_graph, mock_feature_matrix, mock_concept_embeddings, tmp_path,
    ):
        """The .mappings.json should contain structural and clinical maps."""
        import json
        output_path = tmp_path / "mappings_test.pt"
        export_rdf_to_heterodata(
            rdf_graph=export_test_graph,
            feature_matrix_path=mock_feature_matrix,
            concept_embeddings_path=mock_concept_embeddings,
            output_path=output_path,
            embed_unmapped_fn=_mock_embed_unmapped,
        )
        mappings_path = output_path.with_suffix(".mappings.json")
        assert mappings_path.exists()
        data = json.loads(mappings_path.read_text())
        assert "structural" in data
        assert "clinical" in data
        assert len(data["structural"]["patient"]) == 3
        assert len(data["structural"]["admission"]) == 4
        assert len(data["clinical"]["lab"]) == 2
        assert len(data["clinical"]["vital"]) == 1


class TestEmptyClinicalType:
    def test_no_events_of_type(self, tmp_path):
        """A graph with zero events of a clinical type should produce 0-node types."""
        g = initialize_graph(ONTOLOGY_DIR)

        # Minimal graph: 1 patient, 1 admission, no events
        p = write_patient(g, {"subject_id": 1, "gender": "M", "anchor_age": 50})
        write_admission(g, {
            "hadm_id": 1, "subject_id": 1,
            "admittime": datetime(2150, 1, 1, 8, 0, 0),
            "dischtime": datetime(2150, 1, 5, 14, 0, 0),
            "admission_type": "EMERGENCY", "discharge_location": "HOME",
            "readmitted_30d": False, "readmitted_60d": False,
        }, p)

        # Parquet with 1 admission
        df = pd.DataFrame({
            "hadm_id": [1], "subject_id": [1],
            "readmitted_30d": [0], "readmitted_60d": [0],
            "feat_1": [1.0],
        })
        feat_path = tmp_path / "feat.parquet"
        df.to_parquet(feat_path, index=False)

        # Empty concept embeddings
        emb_path = tmp_path / "emb.pt"
        torch.save({}, emb_path)

        output_path = tmp_path / "empty_test.pt"
        data = export_rdf_to_heterodata(
            rdf_graph=g,
            feature_matrix_path=feat_path,
            concept_embeddings_path=emb_path,
            output_path=output_path,
            embed_unmapped_fn=_mock_embed_unmapped,
        )

        assert data["patient"].num_nodes == 1
        assert data["admission"].num_nodes == 1
        # All clinical types should be 0
        for ntype in ("lab", "vital", "drug", "diagnosis", "microbiology"):
            assert data[ntype].num_nodes == 0
