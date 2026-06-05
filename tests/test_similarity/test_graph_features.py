"""Tests for the graph-derived similarity feature extractors (plan III-A).

The fixture builds a small RDF graph using the *exact* predicates the
production writers emit (verified against ``event_writers.py`` /
``patient_writer.py``), so a template that drifts from the writer schema
fails here rather than silently returning empty columns on the live graph.

Three admissions with deterministic trajectories:

  hadm 100 — lactate rising (2 -> 4 -> 6), vasopressor dose rising (4 -> 8 -> 12),
             two distinct drugs, 72h ICU LOS, a lactate->pressor precedence edge.
  hadm 101 — lactate falling (5 -> 3 -> 1), a single pressor dose (degenerate
             trend), 48h ICU LOS.
  hadm 102 — no lactate at all (absent -> NaN), 24h ICU LOS.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from rdflib import RDF, Graph, Literal, URIRef
from rdflib.namespace import XSD

from src.graph_construction.ontology import MIMIC_NS, TIME_NS
from src.pygower import Kind
from src.similarity.graph_features import (
    GraphFeatureRequest,
    TEMPLATES,
    extract_graph_features,
    feature_kinds,
)

_T0 = datetime(2150, 1, 1, 0, 0, 0)


def _ts(dt: datetime) -> Literal:
    """A dateTimeStamp literal in the exact format ``_write_instant`` emits."""
    return Literal(dt.strftime("%Y-%m-%dT%H:%M:%S") + "Z", datatype=XSD.dateTimeStamp)


def _instant(g: Graph, uri: URIRef, dt: datetime) -> None:
    g.add((uri, RDF.type, TIME_NS.Instant))
    g.add((uri, TIME_NS.inXSDDateTimeStamp, _ts(dt)))


def _admission(g: Graph, hadm_id: int) -> URIRef:
    adm = MIMIC_NS[f"HA-{hadm_id}"]
    g.add((adm, RDF.type, MIMIC_NS.HospitalAdmission))
    g.add((adm, MIMIC_NS.hasAdmissionId, Literal(hadm_id, datatype=XSD.integer)))
    return adm


def _icu_stay(
    g: Graph, adm: URIRef, stay_id: int, intime: datetime, outtime: datetime | None
) -> URIRef:
    stay = MIMIC_NS[f"IS-{stay_id}"]
    g.add((stay, RDF.type, MIMIC_NS.ICUStay))
    g.add((stay, MIMIC_NS.hasStayId, Literal(stay_id, datatype=XSD.integer)))
    begin = MIMIC_NS[f"ICUStayBegin_{stay_id}"]
    _instant(g, begin, intime)
    g.add((stay, TIME_NS.hasBeginning, begin))
    if outtime is not None:
        end = MIMIC_NS[f"ICUStayEnd_{stay_id}"]
        _instant(g, end, outtime)
        g.add((stay, TIME_NS.hasEnd, end))
    g.add((adm, MIMIC_NS.containsICUStay, stay))
    return stay


def _local(stay: URIRef) -> str:
    return str(stay).rsplit("#", 1)[-1]


def _biomarker(
    g: Graph, stay: URIRef, label: str, value: float, at: datetime
) -> URIRef:
    slug = at.strftime("%Y%m%d%H%M%S")
    uri = MIMIC_NS[f"BME-{_local(stay)}-{label}-{slug}"]
    g.add((uri, RDF.type, MIMIC_NS.BioMarkerEvent))
    g.add((uri, RDF.type, TIME_NS.Instant))
    g.add((uri, MIMIC_NS.associatedWithICUStay, stay))
    g.add((uri, MIMIC_NS.hasBiomarkerType, Literal(label, datatype=XSD.string)))
    g.add((uri, MIMIC_NS.hasValue, Literal(value, datatype=XSD.decimal)))
    g.add((uri, TIME_NS.inXSDDateTimeStamp, _ts(at)))
    return uri


def _prescription(
    g: Graph,
    stay: URIRef,
    drug: str,
    category: str | None,
    dose: float | None,
    start: datetime,
) -> URIRef:
    slug = start.strftime("%Y%m%d%H%M%S")
    uri = MIMIC_NS[f"RXE-{_local(stay)}-{drug}-{slug}"]
    g.add((uri, RDF.type, MIMIC_NS.PrescriptionEvent))
    g.add((uri, RDF.type, TIME_NS.ProperInterval))
    g.add((uri, MIMIC_NS.associatedWithICUStay, stay))
    g.add((uri, MIMIC_NS.hasDrugName, Literal(drug, datatype=XSD.string)))
    if category is not None:
        g.add((uri, MIMIC_NS.hasDrugCategory, Literal(category, datatype=XSD.string)))
    if dose is not None:
        g.add((uri, MIMIC_NS.hasDoseValue, Literal(dose, datatype=XSD.decimal)))
    begin = MIMIC_NS[f"RXEBegin-{_local(stay)}-{drug}-{slug}"]
    _instant(g, begin, start)
    g.add((uri, TIME_NS.hasBeginning, begin))
    return uri


@pytest.fixture
def cohort_graph() -> Graph:
    g = Graph()

    # -- hadm 100: rising lactate, rising pressor dose, two drugs ------------
    a100 = _admission(g, 100)
    s100 = _icu_stay(g, a100, 500, _T0, _T0 + timedelta(hours=72))
    lac0 = _biomarker(g, s100, "Lactate", 2.0, _T0 + timedelta(hours=2))
    _biomarker(g, s100, "Lactate", 4.0, _T0 + timedelta(hours=26))
    _biomarker(g, s100, "Lactate", 6.0, _T0 + timedelta(hours=50))
    _prescription(g, s100, "Norepinephrine", "vasopressors", 4.0, _T0 + timedelta(hours=1))
    _prescription(g, s100, "Norepinephrine", "vasopressors", 8.0, _T0 + timedelta(hours=13))
    pre0 = _prescription(g, s100, "Norepinephrine", "vasopressors", 12.0, _T0 + timedelta(hours=25))
    _prescription(g, s100, "Vancomycin", "antibiotics", 1000.0, _T0 + timedelta(hours=3))
    # Allen precedence edge: first lactate before first pressor.
    g.add((lac0, TIME_NS.before, pre0))

    # -- hadm 101: falling lactate, single pressor dose ---------------------
    a101 = _admission(g, 101)
    s101 = _icu_stay(g, a101, 501, _T0, _T0 + timedelta(hours=48))
    _biomarker(g, s101, "Lactate", 5.0, _T0 + timedelta(hours=1))
    _biomarker(g, s101, "Lactate", 3.0, _T0 + timedelta(hours=13))
    _biomarker(g, s101, "Lactate", 1.0, _T0 + timedelta(hours=25))
    _prescription(g, s101, "Norepinephrine", "vasopressors", 6.0, _T0 + timedelta(hours=2))

    # -- hadm 102: no lactate, one drug, 24h ICU LOS -----------------------
    a102 = _admission(g, 102)
    s102 = _icu_stay(g, a102, 502, _T0, _T0 + timedelta(hours=24))
    _prescription(g, s102, "Aspirin", None, None, _T0 + timedelta(hours=4))

    return g


# ---------------------------------------------------------------------------
# series template — biomarker trajectory aggregations
# ---------------------------------------------------------------------------


def test_series_slope_sign_and_magnitude(cohort_graph):
    req = GraphFeatureRequest(
        column="lactate_slope", template="sim_series_by_admission",
        concept="Lactate", params={"agg": "slope"},
    )
    out = TEMPLATES["sim_series_by_admission"].fn(cohort_graph, req)
    # rising 2->4->6 over hours 2,26,50: slope = 4 / 48 per hour.
    assert out[100] == pytest.approx(4.0 / 48.0, rel=1e-6)
    # falling 5->3->1 over hours 1,13,25: slope = -4 / 24 per hour.
    assert out[101] == pytest.approx(-4.0 / 24.0, rel=1e-6)
    # hadm 102 has no lactate at all -> absent (not in dict, becomes NaN).
    assert 102 not in out


def test_series_delta_first_to_last(cohort_graph):
    req = GraphFeatureRequest(
        column="lactate_delta", template="sim_series_by_admission",
        concept="Lactate", params={"agg": "delta"},
    )
    out = TEMPLATES["sim_series_by_admission"].fn(cohort_graph, req)
    assert out[100] == pytest.approx(4.0)    # 6 - 2
    assert out[101] == pytest.approx(-4.0)   # 1 - 5


def test_series_window_restricts_to_first_n_hours(cohort_graph):
    req = GraphFeatureRequest(
        column="lactate_slope_24h", template="sim_series_by_admission",
        concept="Lactate", params={"agg": "slope", "window_hours": 24},
    )
    out = TEMPLATES["sim_series_by_admission"].fn(cohort_graph, req)
    # hadm 100: first event at hour 2; window [2, 26] keeps hours 2 and 26
    # only (values 2 and 4): slope = 2 / 24 per hour.
    assert out[100] == pytest.approx(2.0 / 24.0, rel=1e-6)


def test_series_single_point_is_absent_not_flat(cohort_graph):
    # A concept with exactly one reading has no defined trend -> NaN (absent),
    # so the trait's missing policy decides, rather than a spurious 0 slope.
    req = GraphFeatureRequest(
        column="vanc_slope", template="sim_series_by_admission",
        concept="this-concept-does-not-exist", params={"agg": "slope"},
    )
    out = TEMPLATES["sim_series_by_admission"].fn(cohort_graph, req)
    assert out == {}


def test_series_is_case_insensitive(cohort_graph):
    req = GraphFeatureRequest(
        column="lac", template="sim_series_by_admission",
        concept="lactate", params={"agg": "slope"},
    )
    out = TEMPLATES["sim_series_by_admission"].fn(cohort_graph, req)
    assert 100 in out and 101 in out


# ---------------------------------------------------------------------------
# dose_series template — per-administration dose trajectory by drug category
# ---------------------------------------------------------------------------


def test_dose_series_slope_by_category(cohort_graph):
    req = GraphFeatureRequest(
        column="pressor_dose_slope", template="sim_dose_series",
        concept="vasopressors", params={"agg": "slope"},
    )
    out = TEMPLATES["sim_dose_series"].fn(cohort_graph, req)
    # hadm 100: doses 4,8,12 at hours 1,13,25 -> slope = 8 / 24 per hour.
    assert out[100] == pytest.approx(8.0 / 24.0, rel=1e-6)
    # hadm 101: single pressor dose -> degenerate trend -> absent (NaN).
    assert 101 not in out


# ---------------------------------------------------------------------------
# distinct_drug_count, icu_los, time_to_first_event
# ---------------------------------------------------------------------------


def test_distinct_drug_count(cohort_graph):
    req = GraphFeatureRequest(
        column="n_drugs", template="sim_distinct_drug_count", concept=None,
    )
    out = TEMPLATES["sim_distinct_drug_count"].fn(cohort_graph, req)
    assert out[100] == 2    # Norepinephrine + Vancomycin
    assert out[101] == 1    # Norepinephrine only


def test_icu_los_hours(cohort_graph):
    req = GraphFeatureRequest(
        column="icu_los", template="sim_icu_los", concept=None,
    )
    out = TEMPLATES["sim_icu_los"].fn(cohort_graph, req)
    assert out[100] == pytest.approx(72.0)
    assert out[101] == pytest.approx(48.0)
    assert out[102] == pytest.approx(24.0)


def test_time_to_first_event_hours(cohort_graph):
    req = GraphFeatureRequest(
        column="time_to_lactate", template="sim_time_to_first_event",
        concept="Lactate",
    )
    out = TEMPLATES["sim_time_to_first_event"].fn(cohort_graph, req)
    assert out[100] == pytest.approx(2.0)   # first lactate at intime + 2h
    assert out[101] == pytest.approx(1.0)   # first lactate at intime + 1h
    assert 102 not in out                   # no lactate


# ---------------------------------------------------------------------------
# precedence_count — Allen relations between two concepts
# ---------------------------------------------------------------------------


def test_precedence_count_before(cohort_graph):
    req = GraphFeatureRequest(
        column="lac_before_pressor", template="sim_precedence_count",
        concept="Lactate",
        params={"concept_b": "vasopressors", "relation": "before"},
    )
    out = TEMPLATES["sim_precedence_count"].fn(cohort_graph, req)
    assert out[100] == 1
    assert 101 not in out   # no precedence edge


def test_precedence_as_bool(cohort_graph):
    req = GraphFeatureRequest(
        column="lac_before_pressor", template="sim_precedence_count",
        concept="Lactate",
        params={"concept_b": "vasopressors", "relation": "before", "as_bool": True},
    )
    out = TEMPLATES["sim_precedence_count"].fn(cohort_graph, req)
    assert out[100] == 1.0


# ---------------------------------------------------------------------------
# extract_graph_features — typed multi-column assembly
# ---------------------------------------------------------------------------


def test_extract_assembles_indexed_frame(cohort_graph):
    reqs = [
        GraphFeatureRequest("lactate_slope", "sim_series_by_admission", "Lactate", {"agg": "slope"}),
        GraphFeatureRequest("pressor_dose_slope", "sim_dose_series", "vasopressors", {"agg": "slope"}),
        GraphFeatureRequest("n_drugs", "sim_distinct_drug_count", None),
    ]
    df = extract_graph_features(cohort_graph, reqs)
    assert df.index.name == "hadm_id"
    assert list(df.columns) == ["lactate_slope", "pressor_dose_slope", "n_drugs"]
    assert df.loc[100, "lactate_slope"] == pytest.approx(4.0 / 48.0, rel=1e-6)
    # hadm 101 has a single pressor dose -> NaN (absent), column still present.
    assert math.isnan(df.loc[101, "pressor_dose_slope"])


def test_extract_emits_stable_nan_column_for_globally_absent_concept(cohort_graph):
    reqs = [
        GraphFeatureRequest("lactate_slope", "sim_series_by_admission", "Lactate", {"agg": "slope"}),
        GraphFeatureRequest("troponin_slope", "sim_series_by_admission", "Troponin", {"agg": "slope"}),
    ]
    df = extract_graph_features(cohort_graph, reqs)
    # Troponin appears nowhere; the column must still exist, all-NaN, so the
    # fit/transform column set never shifts between cohorts.
    assert "troponin_slope" in df.columns
    assert df["troponin_slope"].isna().all()


def test_extract_empty_requests_returns_empty_frame(cohort_graph):
    df = extract_graph_features(cohort_graph, [])
    assert df.empty


def test_feature_kinds_reports_column_types():
    reqs = [
        GraphFeatureRequest("lactate_slope", "sim_series_by_admission", "Lactate", {"agg": "slope"}),
        GraphFeatureRequest("lac_before_pressor", "sim_precedence_count", "Lactate",
                            {"concept_b": "vasopressors", "as_bool": True}),
    ]
    kinds = feature_kinds(reqs)
    assert kinds["lactate_slope"] == Kind.QUANTITATIVE
    assert kinds["lac_before_pressor"] == Kind.BINARY


def test_unknown_template_raises():
    with pytest.raises(KeyError):
        TEMPLATES["nope"]
    req = GraphFeatureRequest("x", "sim_not_a_template", None)
    with pytest.raises((KeyError, ValueError)):
        extract_graph_features(Graph(), [req])
