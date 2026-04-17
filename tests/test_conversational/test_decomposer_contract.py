"""Contract-level tests for the decomposer.

These tests verify structural invariants and introspective properties that do
NOT depend on a specific LLM response. They are intentionally small in number
and generic in form — each test is a loop over fixtures, not a hand-written
per-case assertion. Adding a new behavioural case means dropping a file into
``fixtures/``, not editing this file.

Phase 0 scope:
  - Every ``json`` block inside the live prompt validates as CompetencyQuestion.
  - Prompt ↔ supported-fields enumeration is consistent (stubbed today, real in Phase 1).
  - ``_validate_return_type`` is idempotent and never raises.
  - Conversation-history truncation is bounded at the last 5 turns and ordered correctly.
  - ``_extract_json`` never raises on arbitrary malformed text.
  - Retry-on-validation-failure produces a correctly shaped follow-up message array.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.conversational.decomposer import (
    _extract_json,
    _validate_return_type,
    decompose,
)
from src.conversational.models import (
    AnswerResult,
    CompetencyQuestion,
    ClinicalConcept,
)
from src.conversational.operations import get_default_registry
from src.conversational.prompts import (
    DECOMPOSITION_SYSTEM_PROMPT,
    build_decomposition_messages,
)

from tests.test_conversational.conftest import (
    load_decomposer_cases,
    load_malformed_json,
    load_prompt_examples,
    mock_anthropic,
)


def _supported_filter_fields() -> frozenset[str]:
    """Source of truth for supported filter fields: the OperationRegistry.

    Pre-Phase-1 this was a literal; now it's a one-liner and the test
    becomes self-updating as new filter ops are registered.
    """
    return get_default_registry().supported_names("filter")


# ---------------------------------------------------------------------------
# 1. Prompt examples are valid CompetencyQuestions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block", load_prompt_examples())
def test_every_prompt_example_validates_as_competency_question(block: dict):
    """Each ```json``` block inside the live prompt must parse and validate.

    This catches prompt edits that introduce malformed JSON or schema drift.
    Runs against the regenerated ``fixtures/prompt_examples/`` snapshot, so the
    live prompt is the source of truth.
    """
    assert not block.get("__parse_error__"), (
        f"Prompt contains an unparseable JSON block:\n{block.get('__raw__')!r}"
    )
    cq = CompetencyQuestion.model_validate(block)
    assert isinstance(cq, CompetencyQuestion)


# ---------------------------------------------------------------------------
# 2. Prompt ↔ supported-fields round-trip
# ---------------------------------------------------------------------------


def test_every_supported_filter_field_appears_in_prompt():
    """Every filter field the decomposer enforces must be named in the prompt.

    Sourced from the ``OperationRegistry`` — self-updating. Register a new
    filter operation and this test fails until the prompt is updated.
    """
    for field in _supported_filter_fields():
        assert field in DECOMPOSITION_SYSTEM_PROMPT, (
            f"Supported filter field {field!r} is missing from the prompt — "
            "the LLM will never be told it's an option."
        )


def test_prompt_does_not_advertise_unsupported_filter_fields():
    """Catch the reverse drift: a name in the prompt's Supported Operations
    section that the decomposer would reject.

    The "Supported Operations" section is registry-rendered; its names come
    straight from ``describe_for_prompt``. Any name that shows up in the
    filter sub-section must therefore be a registered filter — if the prompt
    ever grows a non-registry section advertising extra filters, this test
    trips.
    """
    # Locate the filter sub-section and collect every name appearing at the
    # start of a row (the ``describe_for_prompt`` format is
    # ``  <name:<22> (ops) ...``).
    section_start = DECOMPOSITION_SYSTEM_PROMPT.find(
        "## filter  (used in patient_filters[].field)"
    )
    assert section_start >= 0, "filter sub-section missing from prompt"
    next_section = DECOMPOSITION_SYSTEM_PROMPT.find(
        "## ", section_start + 1
    )
    section = DECOMPOSITION_SYSTEM_PROMPT[section_start:next_section]

    advertised = set(re.findall(r"^\s{2}(\w[\w_]*)\b", section, flags=re.MULTILINE))
    extra = advertised - _supported_filter_fields()
    assert not extra, (
        f"Prompt advertises filter fields that the decomposer will reject: {extra}"
    )


# ---------------------------------------------------------------------------
# 3. _validate_return_type invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", load_decomposer_cases())
def test_validate_return_type_is_idempotent(case: dict):
    """Applying ``_validate_return_type`` twice yields the same result as once.

    Any post-processing rule that is not idempotent is a latent bug: rerunning
    the validator (which can happen in retry loops) must not keep mutating state.
    """
    cq = CompetencyQuestion.model_validate(case["expected_cq"])
    once = _validate_return_type(cq.model_copy(deep=True))
    twice = _validate_return_type(once.model_copy(deep=True))
    assert twice.model_dump() == once.model_dump()


@pytest.mark.parametrize("case", load_decomposer_cases())
def test_validate_return_type_never_raises(case: dict):
    """``_validate_return_type`` must never raise — it's a best-effort cleanup."""
    cq = CompetencyQuestion.model_validate(case["expected_cq"])
    _validate_return_type(cq)


# ---------------------------------------------------------------------------
# 4. Conversation history: truncation and ordering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_turns", [0, 1, 5, 10])
def test_conversation_history_truncation_and_ordering(n_turns: int):
    """The messages array contains the last 5 history turns (user+assistant pairs) + the new question.

    Parametrized over four sizes to exercise (a) empty history, (b) a single
    turn, (c) exactly the truncation boundary, (d) well over the boundary.
    We assert only on roles, counts, and ordering — never on substrings of
    prompt content, so that prompt-text changes don't break the test.
    """
    history = []
    for i in range(n_turns):
        cq = CompetencyQuestion(original_question=f"prev question {i}")
        answer = AnswerResult(text_summary=f"prev answer {i}")
        history.append((cq, answer))

    messages = build_decomposition_messages(
        "new question",
        conversation_history=history or None,
    )

    # At most the last 5 turns (user+assistant = 2 messages each) + the new user message.
    expected_history_turns = min(n_turns, 5)
    assert len(messages) == (expected_history_turns * 2) + 1

    # Roles alternate user/assistant across the history, then a trailing user.
    for idx in range(expected_history_turns):
        assert messages[idx * 2]["role"] == "user"
        assert messages[idx * 2 + 1]["role"] == "assistant"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "new question"

    # History is the tail of the provided list, not the head — verify ordering.
    if n_turns > expected_history_turns:
        # The first included user message should be the (n_turns - 5)-th original.
        expected_first = f"prev question {n_turns - expected_history_turns}"
        assert messages[0]["content"] == expected_first


# ---------------------------------------------------------------------------
# 5. JSON extraction is total (never raises)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", load_malformed_json())
def test_extract_json_never_raises(raw: str):
    """``_extract_json`` is a pure text utility and must be total.

    Callers downstream may still fail to parse the result — that's fine —
    but the extractor itself must never raise. This guarantees the decomposer
    can always fall back to its retry path.
    """
    result = _extract_json(raw)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 6. Retry-on-malformed-JSON: structural assertions
# ---------------------------------------------------------------------------


def test_retry_on_malformed_json_sends_corrective_follow_up():
    """After a malformed first response, the second API call must include
    (a) the prior assistant turn verbatim and (b) a corrective user turn.

    We assert on message roles, counts, and the presence of the prior
    assistant text — NOT on specific wording of the corrective prompt, so
    rewording the retry message doesn't break this test.
    """
    valid = json.dumps({
        "original_question": "avg creatinine",
        "clinical_concepts": [
            {"name": "creatinine", "concept_type": "biomarker"}
        ],
        "return_type": "text",
        "scope": "cohort",
    })
    bad_first_response = "not valid json at all {{{"
    client = mock_anthropic([bad_first_response, valid])

    decompose(client, "test question")

    assert client.messages.create.call_count == 2

    second_call = client.messages.create.call_args_list[1]
    messages = second_call.kwargs["messages"]

    # First call had only the user question. Second call must extend that
    # with the prior assistant turn + a corrective user turn.
    roles = [m["role"] for m in messages]
    assert roles[0] == "user"
    assert "assistant" in roles, "Retry must include the prior assistant turn"

    assistant_turn = next(m for m in messages if m["role"] == "assistant")
    assert assistant_turn["content"] == bad_first_response, (
        "The assistant's failing response must be replayed verbatim so the "
        "LLM can see what it produced and correct itself."
    )

    # The final message is the corrective user prompt.
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] != "test question", (
        "The corrective user turn must differ from the original question."
    )


def test_retry_on_unsupported_filter_sends_corrective_follow_up():
    """When the LLM returns a valid CQ that uses a rejected filter field, the
    decomposer retries with a corrective message naming the unsupported fields.

    We check the retry shape, not its exact wording.
    """
    bad = json.dumps({
        "original_question": "neuro ICU lactate",
        "clinical_concepts": [
            {"name": "lactate", "concept_type": "biomarker"}
        ],
        "patient_filters": [
            {"field": "icu_type", "operator": "=", "value": "neuro"}
        ],
        "return_type": "text",
        "scope": "cohort",
    })
    good = json.dumps({
        "original_question": "neuro ICU lactate",
        "clinical_concepts": [
            {"name": "lactate", "concept_type": "biomarker"}
        ],
        "patient_filters": [
            {"field": "diagnosis", "operator": "contains", "value": "neuro"}
        ],
        "return_type": "text",
        "scope": "cohort",
    })
    client = mock_anthropic([bad, good])

    result = decompose(client, "lactate for neuro ICU patients")

    assert client.messages.create.call_count == 2
    # Final result must only reference supported fields.
    for f in result.patient_filters:
        assert f.field in _supported_filter_fields()


# ---------------------------------------------------------------------------
# 7. Prompt examples round-trip through CompetencyQuestion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block", load_prompt_examples())
def test_prompt_examples_validate_return_type_idempotent(block: dict):
    """Combine (1) and (3): every prompt example, when parsed, survives
    ``_validate_return_type`` idempotently. Catches the case where a prompt
    example itself would be 'corrected' on first run — a sign the example
    teaches the LLM something the validator will then undo.
    """
    if block.get("__parse_error__"):
        pytest.skip("Unparseable prompt block — covered by validity test")
    cq = CompetencyQuestion.model_validate(block)
    once = _validate_return_type(cq.model_copy(deep=True))
    twice = _validate_return_type(once.model_copy(deep=True))
    assert twice.model_dump() == once.model_dump()


# ---------------------------------------------------------------------------
# 8. Phase 3 — self-awareness, pipeline staging, registry-driven sections
# ---------------------------------------------------------------------------


class TestPipelineSelfAwareness:
    """The prompt must tell the LLM where it sits in the pipeline, so that
    downstream-step-specific failure modes (LLM writing SQL, inventing data)
    can be avoided. These tests enforce the role section is present and
    names every pipeline stage."""

    def test_role_section_declares_decomposer(self):
        assert "Decomposer" in DECOMPOSITION_SYSTEM_PROMPT

    @pytest.mark.parametrize("stage", [
        "Decompose", "Extract", "Graph build", "Reason", "Answer",
    ])
    def test_every_pipeline_stage_named_in_prompt(self, stage: str):
        assert stage in DECOMPOSITION_SYSTEM_PROMPT, (
            f"Pipeline stage {stage!r} missing from the Role section — "
            "self-awareness is incomplete."
        )

    def test_role_section_forbids_sql_generation(self):
        """The LLM must be told not to write SQL. This is the single most
        important constraint — a regression here leaks compute into step 1."""
        assert "NEVER" in DECOMPOSITION_SYSTEM_PROMPT
        assert "SQL" in DECOMPOSITION_SYSTEM_PROMPT


class TestRegistryDrivenSections:
    """Every registered operation name across all three kinds must appear in
    its own prompt sub-section. Aggregate and comparison_axis dispatch is
    new in Phase 3 — Phase 1b already covers filter fields."""

    @pytest.mark.parametrize("kind,subsection_header", [
        ("filter", "## filter"),
        ("aggregate", "## aggregate"),
        ("comparison_axis", "## comparison_axis"),
    ])
    def test_every_registered_name_appears_in_its_subsection(
        self, kind: str, subsection_header: str,
    ):
        from src.conversational.operations import get_default_registry

        start = DECOMPOSITION_SYSTEM_PROMPT.find(subsection_header)
        assert start >= 0, f"subsection {subsection_header!r} missing from prompt"
        next_header = DECOMPOSITION_SYSTEM_PROMPT.find("## ", start + 1)
        # If this is the last subsection, take everything to the next top-level
        # ``# `` header; otherwise stop at the next ``## `` sub-header.
        if next_header < 0:
            end = DECOMPOSITION_SYSTEM_PROMPT.find("\n# ", start + 1)
            next_header = end if end >= 0 else len(DECOMPOSITION_SYSTEM_PROMPT)
        subsection = DECOMPOSITION_SYSTEM_PROMPT[start:next_header]

        for name in get_default_registry().supported_names(kind):
            assert name in subsection, (
                f"Operation name {name!r} (kind={kind}) missing from the "
                f"{subsection_header} sub-section"
            )


class TestPromptSnapshot:
    """The committed ``prompts_snapshot.txt`` is the rendered prompt verbatim.
    When the prompt changes — because someone added an example, registered
    a new operation, edited a static section, whatever — this test fails
    loudly and forces the snapshot to be regenerated in the same commit.
    That way prompt changes are visible in PR diffs rather than buried in
    a function call that nobody reads."""

    SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "prompts_snapshot.txt"

    def test_prompt_matches_snapshot(self):
        if not self.SNAPSHOT_PATH.exists():
            self.SNAPSHOT_PATH.write_text(DECOMPOSITION_SYSTEM_PROMPT)
            pytest.skip(
                f"Snapshot written to {self.SNAPSHOT_PATH}. Re-run tests to verify."
            )
        expected = self.SNAPSHOT_PATH.read_text()
        if expected != DECOMPOSITION_SYSTEM_PROMPT:
            diff_preview = (
                f"Prompt changed. To accept, re-run with:\n"
                f"    rm {self.SNAPSHOT_PATH}\n"
                f"    pytest {__file__}\n"
                f"Length: expected {len(expected)} chars, got "
                f"{len(DECOMPOSITION_SYSTEM_PROMPT)}."
            )
            assert expected == DECOMPOSITION_SYSTEM_PROMPT, diff_preview


class TestBuildSystemPromptPluggable:
    """``build_system_prompt(registry)`` must accept an arbitrary registry so
    tests can exercise the prompt with registries they control — and so the
    Phase-5 SNOMED work can swap the registry when a resolver is present.
    """

    def test_builder_uses_supplied_registry(self):
        from src.conversational.operations import (
            FilterFragment,
            FilterOperation,
            OperationRegistry,
        )
        from src.conversational.prompts import build_system_prompt

        custom = OperationRegistry()
        custom.register(FilterOperation(
            name="zzz_custom_field",
            operators=frozenset({"="}),
            value_type="scalar",
            description="test-only filter registered in a custom registry",
            compile_fn=lambda f, ctx: FilterFragment(),
        ))
        prompt = build_system_prompt(custom)
        assert "zzz_custom_field" in prompt
        # Default-registry names should NOT appear because we passed a
        # fresh registry containing only the custom op.
        assert "readmitted_30d" not in prompt.split("## filter")[1].split("## ")[0]


# ---------------------------------------------------------------------------
# 9. Example files live in their canonical location
# ---------------------------------------------------------------------------


class TestPromptExampleLocation:
    """Phase 3 inverted the source of truth: example JSON files under
    ``src/conversational/prompt_examples/single_cq/`` are what gets rendered
    into the prompt, not a prompt string that we then extract from. This
    test guards against a regression where those files get ignored."""

    def test_single_cq_directory_exists_and_is_nonempty(self):
        from src.conversational.prompts import PROMPT_EXAMPLES_DIR

        single_dir = PROMPT_EXAMPLES_DIR / "single_cq"
        assert single_dir.exists(), (
            f"Source-of-truth examples directory missing: {single_dir}"
        )
        files = list(single_dir.glob("*.json"))
        assert files, "No single_cq examples on disk — prompt will be empty"

    def test_every_source_example_appears_verbatim_in_prompt(self):
        """Structural guard: the rendered prompt includes each source
        example's ``original_question`` so the LLM sees them as worked
        examples, not just loose JSON."""
        from src.conversational.prompts import (
            DECOMPOSITION_SYSTEM_PROMPT,
            PROMPT_EXAMPLES_DIR,
        )

        single_dir = PROMPT_EXAMPLES_DIR / "single_cq"
        for path in sorted(single_dir.glob("*.json")):
            data = json.loads(path.read_text())
            q = data.get("original_question", "")
            assert q and q in DECOMPOSITION_SYSTEM_PROMPT, (
                f"Example {path.name!r} with question {q!r} did not make it "
                f"into the rendered prompt."
            )


# ---------------------------------------------------------------------------
# 10. Phase 4 — interpretation echo + clarifying-question
# ---------------------------------------------------------------------------


class TestPhase4PromptSections:
    """The two Phase-4 prompt sections must name their trigger conditions and
    checklists so the LLM actually follows them. Structural guards only —
    no substring matching on prose that might get reworded."""

    @pytest.mark.parametrize("section_header", [
        "# When to Ask a Clarifying Question",
        "# Self-check Before Responding",
    ])
    def test_section_present_in_prompt(self, section_header: str):
        from src.conversational.prompts import DECOMPOSITION_SYSTEM_PROMPT

        assert section_header in DECOMPOSITION_SYSTEM_PROMPT, (
            f"Section {section_header!r} missing from prompt"
        )

    def test_output_schema_lists_new_fields(self):
        """Phase 4 adds interpretation_summary (required) and clarifying_question
        (optional) to the output schema. The LLM needs both named in the
        schema block or it won't emit them."""
        from src.conversational.prompts import DECOMPOSITION_SYSTEM_PROMPT

        schema_start = DECOMPOSITION_SYSTEM_PROMPT.find("# Output Schema")
        schema_end = DECOMPOSITION_SYSTEM_PROMPT.find("# ", schema_start + 1)
        schema_block = DECOMPOSITION_SYSTEM_PROMPT[schema_start:schema_end]
        assert "interpretation_summary" in schema_block
        assert "clarifying_question" in schema_block


class TestInterpretationSynthesis:
    """When the LLM omits interpretation_summary, the decomposer fills in a
    mechanical restatement so the clinician-facing echo is never blank."""

    def test_llm_omits_interpretation_decomposer_synthesises_one(self):
        from src.conversational.decomposer import decompose

        # Mock returns a CQ JSON without interpretation_summary.
        cq_json = json.dumps({
            "original_question": "avg creatinine over 65",
            "clinical_concepts": [
                {"name": "creatinine", "concept_type": "biomarker"},
            ],
            "patient_filters": [
                {"field": "age", "operator": ">", "value": "65"},
            ],
            "aggregation": "mean",
            "return_type": "text",
            "scope": "cohort",
        })
        client = mock_anthropic([cq_json])
        result = decompose(client, "avg creatinine over 65")

        assert result.interpretation_summary, (
            "Decomposer must synthesise interpretation_summary when the LLM omits it"
        )
        # Structural properties of the synthesised string: mentions the
        # concept name and the aggregation verb.
        assert "creatinine" in result.interpretation_summary.lower()
        assert "mean" in result.interpretation_summary.lower()

    def test_whitespace_only_interpretation_is_treated_as_missing(self):
        """If the LLM returns ``"   "`` we treat it as missing and synthesise."""
        from src.conversational.decomposer import decompose

        cq_json = json.dumps({
            "original_question": "count sepsis",
            "clinical_concepts": [
                {"name": "sepsis", "concept_type": "diagnosis"},
            ],
            "aggregation": "count",
            "return_type": "text",
            "scope": "cohort",
            "interpretation_summary": "   ",
        })
        client = mock_anthropic([cq_json])
        result = decompose(client, "count sepsis")
        assert result.interpretation_summary
        assert result.interpretation_summary.strip()

    def test_llm_supplied_interpretation_is_preserved(self):
        """If the LLM writes a good interpretation, we keep it verbatim —
        no silent rewriting."""
        from src.conversational.decomposer import decompose

        pinned = "Mean creatinine over admissions with anchor_age > 65."
        cq_json = json.dumps({
            "original_question": "avg creatinine over 65",
            "clinical_concepts": [
                {"name": "creatinine", "concept_type": "biomarker"},
            ],
            "patient_filters": [
                {"field": "age", "operator": ">", "value": "65"},
            ],
            "aggregation": "mean",
            "return_type": "text",
            "scope": "cohort",
            "interpretation_summary": pinned,
        })
        client = mock_anthropic([cq_json])
        result = decompose(client, "avg creatinine over 65")
        assert result.interpretation_summary == pinned

    def test_clarifying_question_passes_through_when_set(self):
        """Decomposer must not clobber a clarifying_question produced by the LLM."""
        from src.conversational.decomposer import decompose

        cq_json = json.dumps({
            "original_question": "show me the labs",
            "clinical_concepts": [],
            "return_type": "text_and_table",
            "scope": "cohort",
            "interpretation_summary": "Unclear request.",
            "clarifying_question": "Which labs, for which patients?",
        })
        client = mock_anthropic([cq_json])
        result = decompose(client, "show me the labs")
        assert result.clarifying_question == "Which labs, for which patients?"


class TestSynthesisCoversEveryFixture:
    """Every fixture case, after decomposition, must yield a non-empty
    interpretation_summary — either from the fixture's pinned value or from
    the synthesiser fallback."""

    @pytest.mark.parametrize("case", load_decomposer_cases())
    def test_fixture_yields_nonempty_interpretation(self, case):
        from src.conversational.decomposer import decompose

        client = mock_anthropic([case["llm_response"]])
        result = decompose(client, case["question"])
        assert result.interpretation_summary
        assert result.interpretation_summary.strip()
