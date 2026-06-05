"""End-to-end cohort-by-similarity tests through ``ConversationalPipeline.ask``
(plan III-E).

These drive a free-text clinician request all the way through the live pipeline
— decomposer routing → described-profile detection → the Sonnet definition
builder → criteria logging → the anchorless cohort runner (real Gower over a
synthetic DuckDB) → the ``AnswerResult`` the Streamlit UI renders — with only
the two LLM boundaries mocked:

  * ``decompose_question`` is patched to emit a ``patient_similarity`` CQ with a
    free-text ``anchor_template`` (so the orchestrator takes the
    described-profile branch — the one that logs criteria and calls the
    builder, ``orchestrator.py:1216``), and
  * the pipeline's anthropic client is swapped so ``build_definition`` parses a
    canned ``CohortDefinition`` JSON instead of calling the network.

Everything else is real: the frozen reference ranges (``load_reference_ranges``),
the DuckDB feature pull, the pygower scoring, the QoI/CSV assembly, and — for the
temporal case — the per-question RDF graph built over the candidate pool. Two
shapes are covered:

  * a *contextual-only* cohort (a symmetric ``age`` trait), and
  * a *temporal* cohort whose trait is graph-derived. ICU length-of-stay via the
    ``sim_icu_los`` extractor stands in for the plan's "worsening lactate" slope:
    the synthetic labs are single-reading, so a real slope is undefined, whereas
    ICU LOS is genuine data that still exercises the full
    extract → build_query_graph → extract_graph_features → score path.

The whole of ``ask()`` is wrapped in a swallow-all ``except`` (orchestrator.py:394),
so each test asserts on concrete membership rather than mere absence of an
exception — a silent failure would surface as an empty ``data_table`` whose
``text_summary`` we print in the assertion message.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.conversational.models import CompetencyQuestion, DecompositionResult
from src.conversational.orchestrator import ConversationalPipeline
from src.similarity.models import SimilaritySpec

_ONTOLOGY_DIR = Path(__file__).parent.parent.parent / "ontology" / "definition"


# Single symmetric age trait. Frozen range is (19, 91) ⇒ span 72, so a
# candidate's distance is |age - 68| / 72. Threshold 0.30 keeps every synthetic
# admission but the 45-year-old (105, distance 0.319).
_CONTEXTUAL_JSON = json.dumps({
    "prefilters": [],
    "traits": [
        {"name": "age", "source": "sql", "kind": "quantitative",
         "reference_value": 68, "direction": "symmetric", "weight": 1.0},
    ],
    "distance_threshold": 0.30,
    "top_k": 30,
})

# Graph-derived ICU length-of-stay (hours), scored against the frozen
# icu_los_hours range. ref 70h == stay 1001's LOS, so admission 101 is an exact
# match; 102/104/105 have no ICU stay ⇒ NaN ⇒ excluded.
_TEMPORAL_JSON = json.dumps({
    "prefilters": [],
    "traits": [
        {"name": "icu_los_hours", "source": "graph_temporal",
         "kind": "quantitative", "reference_value": 70,
         "direction": "symmetric", "weight": 1.0,
         "template": "sim_icu_los", "concept": None, "graph_params": {}},
    ],
    "distance_threshold": 0.5,
    "top_k": 30,
})


def _mock_client(cohort_json: str) -> MagicMock:
    """An anthropic stand-in whose ``messages.create`` always returns
    *cohort_json* as a single text block (what ``build_definition`` parses)."""
    client = MagicMock()
    resp = MagicMock()
    block = MagicMock(type="text")
    block.text = cohort_json
    resp.content = [block]
    resp.stop_reason = "end_turn"
    client.messages.create.return_value = resp
    return client


def _similarity_decomp(question: str) -> DecompositionResult:
    """A decomposer result that routes to the described-profile cohort path:
    ``scope='patient_similarity'`` + a free-text ``anchor_template`` (no concrete
    patient anchor, no pre-baked ``cohort_definition``)."""
    cq = CompetencyQuestion(
        original_question=question,
        scope="patient_similarity",
        similarity_spec=SimilaritySpec(anchor_template={"description": question}),
    )
    return DecompositionResult(competency_questions=[cq])


def _make_pipeline(cohort_json: str, tmp_path: Path, monkeypatch) -> ConversationalPipeline:
    """A pipeline with every non-similarity LLM stage disabled, its anthropic
    client swapped for a canned cohort-JSON mock, and the criteria log routed to
    a temp file. ``OMOPHUB_API_KEY`` is cleared so the (real) concept resolver
    stays offline regardless of the developer's environment."""
    monkeypatch.delenv("OMOPHUB_API_KEY", raising=False)
    monkeypatch.setenv("NEUROGRAPH_QUERY_LOG", str(tmp_path / "queries.jsonl"))
    pipeline = ConversationalPipeline(
        db_path=tmp_path / "unused.duckdb",
        ontology_dir=_ONTOLOGY_DIR,
        api_key="test-key",
        data_source="local",
        enable_critic=False,
        enable_pre_validator=False,
        enable_disambiguation=False,
    )
    pipeline._client = _mock_client(cohort_json)
    return pipeline


def _read_criteria(tmp_path: Path) -> list[dict]:
    """The ``cohort_definition`` records written to the JSONL activity log."""
    path = tmp_path / "queries.jsonl"
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in records if r.get("kind") == "cohort_definition"]


@pytest.fixture
def patched_backend(synthetic_duckdb_with_events, monkeypatch):
    """Point the orchestrator's DuckDB backend factory at the already-open
    synthetic connection. This reuses the proven rich fixture and avoids
    DuckDB's single-writer file lock (a second ``connect`` to the same path
    would conflict with the fixture's open handle). ``close`` is a no-op so the
    fixture — not ``ask()`` — owns the connection's lifecycle."""
    from src.conversational.extractor import _DuckDBBackend

    def _factory(_db_path):
        backend = _DuckDBBackend.__new__(_DuckDBBackend)
        backend._conn = synthetic_duckdb_with_events
        backend.close = lambda: None
        return backend

    monkeypatch.setattr(
        "src.conversational.orchestrator._DuckDBBackend", _factory,
    )
    return synthetic_duckdb_with_events


class TestCohortEndToEnd:
    def test_contextual_cohort_through_pipeline_ask(
        self, patched_backend, tmp_path, monkeypatch,
    ):
        question = "find emergency patients like a 68-year-old"
        pipeline = _make_pipeline(_CONTEXTUAL_JSON, tmp_path, monkeypatch)
        with patch(
            "src.conversational.orchestrator.decompose_question",
            return_value=_similarity_decomp(question),
        ):
            answer = pipeline.ask(question)

        # A real cohort came back (not the swallow-all error AnswerResult).
        assert answer.data_table, answer.text_summary
        assert answer.table_columns == ["rank", "hadm_id", "subject_id", "distance"]

        # Membership: |age - 68| / 72 ≤ 0.30 keeps everyone but the 45yo (105).
        hadms = {row["hadm_id"] for row in answer.data_table}
        assert hadms == {101, 102, 103, 104, 106}
        assert 105 not in hadms
        # Threshold honored on every returned member.
        assert all(row["distance"] <= 0.30 for row in answer.data_table)
        # Sorted nearest-first; the 65yo admissions (101/102) lead.
        dists = [row["distance"] for row in answer.data_table]
        assert dists == sorted(dists)
        assert answer.data_table[0]["hadm_id"] in (101, 102)

        # Downloadable (subject_id, hadm_id) take-away.
        assert answer.download_filename == "cohort.csv"
        assert answer.download_csv is not None
        lines = answer.download_csv.strip().splitlines()
        assert lines[0] == "rank,subject_id,hadm_id,distance"
        assert len(lines) - 1 == len(answer.data_table)

        # Criteria logged to the JSONL activity log (reproducible from the log).
        criteria = _read_criteria(tmp_path)
        assert len(criteria) == 1
        assert criteria[0]["question"] == question
        assert criteria[0]["distance_threshold"] == 0.30
        age = next(t for t in criteria[0]["traits"] if t["name"] == "age")
        assert age["source"] == "sql"
        assert age["direction"] == "symmetric"

    def test_temporal_cohort_builds_graph_through_pipeline_ask(
        self, patched_backend, tmp_path, monkeypatch,
    ):
        question = "patients like a long ICU stay with worsening labs"
        pipeline = _make_pipeline(_TEMPORAL_JSON, tmp_path, monkeypatch)
        with patch(
            "src.conversational.orchestrator.decompose_question",
            return_value=_similarity_decomp(question),
        ):
            answer = pipeline.ask(question)

        # The graph-temporal trait drove a real per-question RDF graph build over
        # the pool; a cohort came back (not the swallow-all error AnswerResult).
        assert answer.data_table, answer.text_summary
        hadms = {row["hadm_id"] for row in answer.data_table}
        # Only admissions with an ICU stay get a graph-derived LOS; 102/104/105
        # have none ⇒ NaN ⇒ excluded.
        assert hadms <= {101, 103, 106}
        assert 101 in hadms
        assert {102, 104, 105}.isdisjoint(hadms)
        # ref 70h == stay 1001's LOS ⇒ admission 101 is an exact match.
        m101 = next(r for r in answer.data_table if r["hadm_id"] == 101)
        assert m101["distance"] == pytest.approx(0.0, abs=1e-6)
        assert all(row["distance"] <= 0.5 for row in answer.data_table)

        # CSV present and criteria logged, including the graph-derived source.
        assert answer.download_csv is not None
        criteria = _read_criteria(tmp_path)
        assert len(criteria) == 1
        assert criteria[0]["distance_threshold"] == 0.5
        icu = next(t for t in criteria[0]["traits"] if t["name"] == "icu_los_hours")
        assert icu["source"] == "graph_temporal"
