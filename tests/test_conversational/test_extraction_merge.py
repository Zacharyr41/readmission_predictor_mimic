"""Tests for ``merge_extractions`` — union-with-dedup across ExtractionResults.

Phase 4.5: when a big question decomposes into N CompetencyQuestions, each is
run through the extractor independently, yielding N ExtractionResults. Before
graph construction those are merged into one unified ExtractionResult so
exactly ONE RDF graph is built from the union of all discovered data.

The semantic contract: merging preserves every unique row exactly once. Dedup
keys per concept type are derived from MIMIC-IV identifiers (labevent_id,
microevent_id) where they exist and from composite keys otherwise. RDF triples
are set-valued in the graph builder already, so the merge only has to dedup
the ExtractionResult dicts — it doesn't have to worry about downstream URI
collisions.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.conversational.models import ExtractionResult


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _patient(sid: int) -> dict:
    return {"subject_id": sid, "gender": "M", "anchor_age": 65}


def _admission(hadm_id: int, sid: int) -> dict:
    return {
        "hadm_id": hadm_id,
        "subject_id": sid,
        "admittime": datetime(2150, 1, 1),
        "dischtime": datetime(2150, 1, 5),
        "admission_type": "EMERGENCY",
        "discharge_location": "HOME",
        "hospital_expire_flag": 0,
        "readmitted_30d": 0,
        "readmitted_60d": 0,
    }


def _icu(stay_id: int, hadm_id: int, sid: int) -> dict:
    return {
        "stay_id": stay_id,
        "hadm_id": hadm_id,
        "subject_id": sid,
        "intime": datetime(2150, 1, 1),
        "outtime": datetime(2150, 1, 4),
        "los": 3.0,
    }


def _biomarker(labevent_id: int, hadm_id: int = 101) -> dict:
    return {
        "labevent_id": labevent_id,
        "subject_id": 1,
        "hadm_id": hadm_id,
        "itemid": 50912,
        "charttime": datetime(2150, 1, 2, 6, 0),
        "label": "Creatinine",
        "fluid": "Blood",
        "category": "Chemistry",
        "valuenum": 1.2,
        "valueuom": "mg/dL",
        "ref_range_lower": 0.7,
        "ref_range_upper": 1.3,
    }


def _vital(stay_id: int, itemid: int, when: datetime, val: float) -> dict:
    return {
        "stay_id": stay_id,
        "subject_id": 1,
        "hadm_id": 101,
        "itemid": itemid,
        "charttime": when,
        "label": "Heart Rate",
        "category": "Routine Vital Signs",
        "valuenum": val,
    }


def _drug(hadm_id: int, drug: str, start: datetime) -> dict:
    return {
        "hadm_id": hadm_id,
        "subject_id": 1,
        "drug": drug,
        "starttime": start,
        "stoptime": datetime(2150, 1, 3),
        "dose_val_rx": 1000.0,
        "dose_unit_rx": "mg",
        "route": "IV",
    }


def _diagnosis(hadm_id: int, seq: int, code: str) -> dict:
    return {
        "hadm_id": hadm_id,
        "subject_id": 1,
        "seq_num": seq,
        "icd_code": code,
        "icd_version": 10,
        "long_title": f"Diagnosis {code}",
    }


def _microbio(microevent_id: int) -> dict:
    return {
        "microevent_id": microevent_id,
        "subject_id": 1,
        "hadm_id": 101,
        "charttime": datetime(2150, 1, 2, 12, 0),
        "spec_type_desc": "BLOOD CULTURE",
        "org_name": "STAPHYLOCOCCUS AUREUS",
    }


# ---------------------------------------------------------------------------
# 1. Trivial cases
# ---------------------------------------------------------------------------


class TestMergeTrivialCases:
    def test_empty_list_returns_empty_result(self):
        from src.conversational.extractor import merge_extractions

        merged = merge_extractions([])
        assert merged.patients == []
        assert merged.admissions == []
        assert merged.icu_stays == []
        assert merged.events == {}

    def test_single_input_is_idempotent(self):
        """Merging one ExtractionResult returns a structurally equivalent result.

        Not literally the same object — the caller must be able to mutate the
        merged result without affecting the input — but every row is preserved
        exactly once.
        """
        from src.conversational.extractor import merge_extractions

        er = ExtractionResult(
            patients=[_patient(1)],
            admissions=[_admission(101, 1)],
            icu_stays=[_icu(1001, 101, 1)],
            events={"biomarker": [_biomarker(1)]},
        )
        merged = merge_extractions([er])
        assert merged.patients == er.patients
        assert merged.admissions == er.admissions
        assert merged.icu_stays == er.icu_stays
        assert merged.events == er.events


# ---------------------------------------------------------------------------
# 2. Top-level dedup (patients, admissions, icu_stays)
# ---------------------------------------------------------------------------


class TestMergeTopLevel:
    def test_patients_deduped_by_subject_id(self):
        from src.conversational.extractor import merge_extractions

        a = ExtractionResult(patients=[_patient(1), _patient(2)])
        b = ExtractionResult(patients=[_patient(2), _patient(3)])
        merged = merge_extractions([a, b])
        sids = sorted(p["subject_id"] for p in merged.patients)
        assert sids == [1, 2, 3]

    def test_admissions_deduped_by_hadm_id(self):
        from src.conversational.extractor import merge_extractions

        a = ExtractionResult(admissions=[_admission(101, 1), _admission(102, 1)])
        b = ExtractionResult(admissions=[_admission(102, 1), _admission(103, 2)])
        merged = merge_extractions([a, b])
        hadms = sorted(ad["hadm_id"] for ad in merged.admissions)
        assert hadms == [101, 102, 103]

    def test_icu_stays_deduped_by_stay_id(self):
        from src.conversational.extractor import merge_extractions

        a = ExtractionResult(icu_stays=[_icu(1001, 101, 1)])
        b = ExtractionResult(icu_stays=[_icu(1001, 101, 1), _icu(1002, 102, 1)])
        merged = merge_extractions([a, b])
        stay_ids = sorted(s["stay_id"] for s in merged.icu_stays)
        assert stay_ids == [1001, 1002]

    def test_union_of_disjoint_inputs(self):
        from src.conversational.extractor import merge_extractions

        a = ExtractionResult(
            patients=[_patient(1)],
            admissions=[_admission(101, 1)],
            icu_stays=[_icu(1001, 101, 1)],
        )
        b = ExtractionResult(
            patients=[_patient(2)],
            admissions=[_admission(201, 2)],
            icu_stays=[_icu(2001, 201, 2)],
        )
        merged = merge_extractions([a, b])
        assert len(merged.patients) == 2
        assert len(merged.admissions) == 2
        assert len(merged.icu_stays) == 2


# ---------------------------------------------------------------------------
# 3. Event-type dedup
# ---------------------------------------------------------------------------


class TestMergeEvents:
    def test_biomarker_deduped_by_labevent_id(self):
        from src.conversational.extractor import merge_extractions

        a = ExtractionResult(events={"biomarker": [_biomarker(1), _biomarker(2)]})
        b = ExtractionResult(events={"biomarker": [_biomarker(2), _biomarker(3)]})
        merged = merge_extractions([a, b])
        ids = sorted(e["labevent_id"] for e in merged.events["biomarker"])
        assert ids == [1, 2, 3]

    def test_microbiology_deduped_by_microevent_id(self):
        from src.conversational.extractor import merge_extractions

        a = ExtractionResult(events={"microbiology": [_microbio(1), _microbio(2)]})
        b = ExtractionResult(events={"microbiology": [_microbio(2)]})
        merged = merge_extractions([a, b])
        ids = sorted(e["microevent_id"] for e in merged.events["microbiology"])
        assert ids == [1, 2]

    def test_vital_deduped_by_composite_key(self):
        """Vitals lack a single PK, so we dedup on (stay_id, itemid, charttime).

        Same (stay, item, time) from two extractions is the same measurement.
        """
        from src.conversational.extractor import merge_extractions

        t1 = datetime(2150, 1, 1, 10, 0)
        t2 = datetime(2150, 1, 1, 11, 0)
        a = ExtractionResult(events={"vital": [
            _vital(1001, 220045, t1, 78.0),
            _vital(1001, 220045, t2, 82.0),
        ]})
        b = ExtractionResult(events={"vital": [
            _vital(1001, 220045, t1, 78.0),  # duplicate
            _vital(1002, 220045, t1, 90.0),  # different stay → kept
        ]})
        merged = merge_extractions([a, b])
        keys = sorted(
            (v["stay_id"], v["itemid"], v["charttime"]) for v in merged.events["vital"]
        )
        assert keys == sorted([(1001, 220045, t1), (1001, 220045, t2), (1002, 220045, t1)])

    def test_drug_deduped_by_composite_key(self):
        """Drugs dedup on (hadm_id, drug, starttime). Two different start times
        for the same drug in the same admission are distinct prescriptions."""
        from src.conversational.extractor import merge_extractions

        start1 = datetime(2150, 1, 1, 10, 0)
        start2 = datetime(2150, 1, 2, 10, 0)
        a = ExtractionResult(events={"drug": [_drug(101, "Vancomycin", start1)]})
        b = ExtractionResult(events={"drug": [
            _drug(101, "Vancomycin", start1),  # duplicate
            _drug(101, "Vancomycin", start2),  # different start → kept
            _drug(101, "Ceftriaxone", start1), # different drug → kept
        ]})
        merged = merge_extractions([a, b])
        keys = sorted(
            (d["hadm_id"], d["drug"], d["starttime"]) for d in merged.events["drug"]
        )
        assert keys == sorted([
            (101, "Vancomycin", start1),
            (101, "Vancomycin", start2),
            (101, "Ceftriaxone", start1),
        ])

    def test_diagnosis_deduped_by_hadm_and_seq(self):
        from src.conversational.extractor import merge_extractions

        a = ExtractionResult(events={"diagnosis": [_diagnosis(101, 1, "I639")]})
        b = ExtractionResult(events={"diagnosis": [
            _diagnosis(101, 1, "I639"),  # duplicate
            _diagnosis(101, 2, "E11"),   # different seq
        ]})
        merged = merge_extractions([a, b])
        keys = sorted(
            (d["hadm_id"], d["seq_num"]) for d in merged.events["diagnosis"]
        )
        assert keys == [(101, 1), (101, 2)]


# ---------------------------------------------------------------------------
# 4. Cross-concept merging
# ---------------------------------------------------------------------------


class TestMergeCrossConcept:
    def test_different_concepts_produce_same_keys_union(self):
        """If cq_1 asks for creatinine and cq_2 asks for lactate, the biomarker
        rows are disjoint (different itemids) — merge preserves all of them."""
        from src.conversational.extractor import merge_extractions

        cr = _biomarker(1)
        cr["label"] = "Creatinine"
        lac = _biomarker(2)
        lac["label"] = "Lactate"
        lac["itemid"] = 50813
        a = ExtractionResult(events={"biomarker": [cr]})
        b = ExtractionResult(events={"biomarker": [lac]})
        merged = merge_extractions([a, b])
        labels = sorted(e["label"] for e in merged.events["biomarker"])
        assert labels == ["Creatinine", "Lactate"]

    def test_concept_types_from_different_extractions_merged(self):
        from src.conversational.extractor import merge_extractions

        a = ExtractionResult(events={"biomarker": [_biomarker(1)]})
        b = ExtractionResult(events={"vital": [_vital(1001, 220045, datetime(2150, 1, 1, 10, 0), 80.0)]})
        merged = merge_extractions([a, b])
        assert set(merged.events.keys()) == {"biomarker", "vital"}
        assert len(merged.events["biomarker"]) == 1
        assert len(merged.events["vital"]) == 1


# ---------------------------------------------------------------------------
# 5. Defensive edge cases
# ---------------------------------------------------------------------------


class TestMergeEdgeCases:
    def test_unknown_event_type_fallback(self):
        """If a future extractor emits an event_type we don't know a key for,
        merging must not crash. We fall back to id()-based dedup (which
        preserves every row as distinct — worst-case no dedup, never a crash)."""
        from src.conversational.extractor import merge_extractions

        row = {"something": "unusual", "event_id": 42}
        a = ExtractionResult(events={"unknown_type": [row]})
        merged = merge_extractions([a])
        assert "unknown_type" in merged.events
        assert len(merged.events["unknown_type"]) == 1

    def test_merged_is_independent_of_inputs(self):
        """Mutating the merged result must not affect the originals."""
        from src.conversational.extractor import merge_extractions

        original = ExtractionResult(patients=[_patient(1)])
        merged = merge_extractions([original])
        merged.patients.append(_patient(2))
        assert len(original.patients) == 1