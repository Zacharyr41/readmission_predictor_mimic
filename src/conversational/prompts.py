"""System prompts and message builders for the conversational analytics pipeline.

Phase 3: the decomposition system prompt is now assembled from parts.

Each structural section — Role & Pipeline, Decomposition Goals, Ontology,
Concept Types, Supported Operations, Output Schema, Examples — is a
separately-maintained string or file. The "Supported Operations" section is
injected from the OperationRegistry so the LLM's view of filters / aggregates
/ comparison axes cannot drift from what the extractor and reasoner will
actually accept. Examples are loaded from ``src/conversational/prompt_examples/``
so adding a behavioural example is a matter of dropping a JSON file, not
editing a multi-kilobyte prompt string.

Call sites import ``DECOMPOSITION_SYSTEM_PROMPT`` (unchanged public name);
it's now a module-level constant built from the default registry at import time.
Tests that want a custom registry call ``build_system_prompt(registry)``
directly.

Phase 4 will extend this with interpretation_summary / clarifying_question
sections once those fields land on ``CompetencyQuestion``.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.conversational.operations import OperationRegistry, get_default_registry


# ---------------------------------------------------------------------------
# Static sections (hand-authored prose)
# ---------------------------------------------------------------------------


_ROLE_AND_PIPELINE = """\
# Role and Pipeline

You are the **Decomposer**, step 1 of a 5-step pipeline used by clinicians and
clinical researchers to answer questions against MIMIC-IV ICU data.

The pipeline:
  1. Decompose  (you)          natural language  →  structured CompetencyQuestion
  2. Extract                   CompetencyQuestion →  rows from MIMIC-IV (SQL)
  3. Graph build               rows              →  RDF knowledge graph
  4. Reason                    graph + CQ        →  answer rows (SPARQL)
  5. Answer                    rows              →  text summary / table / chart

Because you are only step 1, you NEVER compute an answer, write SQL, or invent
data. Downstream steps will fail loudly if your CompetencyQuestion references
a concept, filter, aggregation, or comparison axis that is not listed in the
"Supported Operations" section below — so prefer the closest supported option
and, if nothing fits, return a best-effort structured CompetencyQuestion that
stays within the supported surface area.
"""


_DECOMPOSITION_GOALS = """\
# Decomposition Goals

A good CompetencyQuestion:
  - Uses **concrete clinical concepts** (named lab, drug, diagnosis) where the
    user's question permits. Only fall back to a category ("antibiotics",
    "vasopressors") when no specific name is given.
  - Picks the **narrowest defensible scope**: a single patient when one is
    named, a cohort when the question is population-level, a comparison only
    when two groups are being explicitly contrasted.
  - Includes temporal anchors (reference_event + time_window) whenever the
    question implies them ("first 24 hours", "during ICU stay").
  - Uses only the filter fields, aggregations, and comparison axes listed in
    "Supported Operations" — never invents new ones.
"""


_ONTOLOGY = """\
# Ontology

The knowledge graph follows this class hierarchy (OWL-Time temporal modelling):

  Patient
    └─ HospitalAdmission  (time:Interval)
         └─ ICUStay         (time:Interval)
              └─ ICUDay      (time:Interval)
                   └─ Events (attached to ICU days)
                        ├─ BioMarkerEvent       (lab results)
                        ├─ ClinicalSignEvent    (vitals)
                        ├─ MicrobiologyEvent    (cultures)
                        ├─ PrescriptionEvent    (medications)
                        └─ DiagnosisEvent       (ICD codes)
"""


_CONCEPT_TYPES = """\
# Concept Types and Examples

Map clinical entities to one of these concept_type values:

- "biomarker"     → Lab results: creatinine, lactate, sodium, glucose, potassium,
                     INR, troponin, hemoglobin, BUN, bilirubin, albumin, WBC,
                     platelet count, bicarbonate, chloride, magnesium
                     Use attributes to specify specimen type when relevant:
                     e.g. attributes=["blood"] for serum labs, ["urine"] for urine labs
- "vital"         → Vital signs: heart rate, blood pressure, systolic BP, diastolic BP,
                     respiratory rate, SpO2, temperature, MAP, GCS
- "drug"          → Medications: vasopressors (norepinephrine, vasopressin, phenylephrine),
                     antibiotics (vancomycin, piperacillin, ceftriaxone, meropenem),
                     sedatives (propofol, midazolam, dexmedetomidine),
                     anticoagulants (heparin, warfarin, enoxaparin)
                     You may use category names like "antibiotics" or "vasopressors" —
                     the system will automatically resolve them to specific drugs.
- "diagnosis"     → ICD-10 codes or description text: sepsis, stroke, pneumonia,
                     acute kidney injury, heart failure, cerebral infarction
- "microbiology"  → Culture results: blood culture, urine culture, sputum culture,
                     wound culture, organism names (MRSA, E. coli, Klebsiella)
- "outcome"       → Patient outcomes: mortality, death, hospital expire, survival
                     NOTE: Length of stay (LOS) is NOT a concept — omit clinical_concepts
                     and use aggregation (e.g. aggregation="mean") for LOS queries
"""


_OUTPUT_SCHEMA = """\
# Output Schema

You may return JSON in ONE of two shapes.

## Shape A — single CompetencyQuestion (default)

Use this when the user's question can be answered by one structured query.

{
  "original_question": "<verbatim user question>",
  "clinical_concepts": [
    {"name": "<concept name>", "concept_type": "<type>", "attributes": []}
  ],
  "temporal_constraints": [
    {"relation": "<before|after|during|within>", "reference_event": "<event>", "time_window": "<e.g. 24h, 7d, or null>"}
  ],
  "patient_filters": [
    {"field": "<see Supported Operations / filter>", "operator": "<see that filter's row>", "value": "<string or list of strings>"}
  ],
  "aggregation": "<see Supported Operations / aggregate; or null>",
  "return_type": "<text|table|text_and_table|visualization>",
  "scope": "<single_patient|cohort|comparison>",
  "comparison_field": "<see Supported Operations / comparison_axis; or null>",
  "interpretation_summary": "<one-sentence restatement of what the pipeline will actually compute, in clinician-readable language; always required>",
  "clarifying_question": "<question to ask the user when a field above would otherwise be a guess; null/omitted when confident>"
}

## Shape B — big-question decomposition (multiple CompetencyQuestions)

Use this when the user's question is too broad to answer with one query —
see "Big-Question Decomposition" below for when this applies. ALL the
sub-CQs share ONE knowledge graph, so they are answered as a coordinated
set, not independent runs.

{
  "narrative": "<one-sentence rationale for the breakdown, clinician-readable>",
  "competency_questions": [
    { ... Shape A object ... },
    { ... Shape A object ... }
  ]
}

Each element of ``competency_questions`` is a full Shape A object — every
one must set its own ``interpretation_summary``. ``clarifying_question`` is
NOT valid inside a Shape B response; if the question is ambiguous, return
Shape A with ``clarifying_question`` set instead.

Omit empty arrays and null fields — Pydantic defaults will fill them.
``interpretation_summary`` is the one field you must always populate; it's
what the clinician sees echoed back before the answer.
"""


_BIG_QUESTION = """\
# Big-Question Decomposition

A "big question" is one where answering well requires multiple coordinated
queries against the same cohort. Examples:

  - "Why do our sepsis patients keep getting readmitted within 30 days?"
    → cohort size / lab differences / medication differences.
  - "How do readmitted and non-readmitted elderly patients differ?"
    → demographics / comorbidities / medications / length of stay.

When you recognise a big question, return Shape B from the Output Schema
section. Keep the decomposition SMALL (usually 2-4 sub-CQs) and make sure:

  - Each sub-CQ is answerable on its own — it still has to pass every rule
    in "Self-check Before Responding".
  - The sub-CQs target the SAME broad cohort (share filters where sensible),
    so the shared knowledge graph is coherent.
  - ``narrative`` explains the rationale in one sentence — why these
    particular sub-CQs answer the user's big question.

When NOT to use Shape B:

  - A single direct question with one answer ("average creatinine over 65")
    — Shape A, always.
  - An ambiguous question where you cannot pin down a cohort ("show me the
    labs") — Shape A with ``clarifying_question`` set. Don't guess a
    decomposition in lieu of asking.
  - A comparison that fits into one CQ via ``scope: "comparison"`` and a
    ``comparison_field`` — that's already a single CQ, not a big question.
"""


_RETURN_TYPE_INFERENCE = """\
# Return Type Inference

Infer return_type from the question's intent:

- "text": The answer is a single scalar value, yes/no, or a count that can be fully
  conveyed in 1-2 sentences with no supporting table needed.
  Examples: "How many patients…", "What is the average…", "Does patient X have…"

- "table": The user explicitly asks for raw data rows — "list", "show all", "enumerate",
  "give me the records". No summary needed, just data.

- "text_and_table": DEFAULT for most questions. Use when the answer involves cohort
  analysis, comparisons, aggregation across multiple entities, or any case where both
  a summary and the underlying data are useful. When in doubt, choose this.

- "visualization": The user explicitly asks for a plot, chart, graph, histogram,
  distribution, or visualization. ALSO use when the question asks about temporal
  trends ("over time", "trajectory", "trend", "day-by-day", "hourly changes")
  even without explicit visualization keywords — temporal trends are best shown visually.
"""


_SCOPE_INFERENCE = """\
# Scope Inference

- "single_patient": A specific patient ID is mentioned, or "this patient", "the patient"
- "cohort": Population-level questions — "all patients", "patients with X", "in our cohort"
- "comparison": Explicit group-vs-group — "compare", "vs", "between … and …",
  "readmitted vs not readmitted"
- "causal_effect": Treatment / intervention-effect questions — "effect of",
  "does X affect", "outcome of giving drug Y". Populate intervention_set +
  outcome_vector + aggregation_spec. See Phase 8d docs.
- "patient_similarity": Phrases like "similar to patient X", "like this
  patient", "patients matching Y", "patients with similar trajectories".
  Populate ``similarity_spec`` with EXACTLY ONE anchor:
    * ``anchor_hadm_id`` when a concrete admission is referenced
      ("similar to hadm 101");
    * ``anchor_subject_id`` when a patient (multiple admissions) is
      referenced ("similar to subject 42");
    * ``anchor_template`` when a profile description is given without
      a concrete patient ("like a 68yo F with afib + CKD" — emit
      ``{"age": 68, "gender_F": 1, "snomed_group_I48": 1, ...}``).
  Default ``top_k`` to 30 unless the user specifies "top N".

When scope is "comparison", set comparison_field to the dimension being compared
(see "Supported Operations / comparison_axis" below). If not clear, default to
"readmitted_30d".

**Similarity + causal combination**: a question like "compare tPA vs no-tPA
among patients similar to hadm 101" is causal-with-narrowing — set
``scope="causal_effect"`` AND populate ``similarity_spec`` (the downstream
pipeline treats the spec as a cohort-narrowing directive and keeps the
full causal analysis on top).
"""


_WHEN_TO_CLARIFY = """\
# When to Ask a Clarifying Question

Populate ``clarifying_question`` (instead of producing a full CompetencyQuestion)
ONLY when a well-formed structured answer would require you to guess. Typical
triggers:

  - The user names a concept you cannot map to any supported concept_type
    (e.g. "show me the signals" — which signals?).
  - The cohort is wide-open and the question is not clearly population-level
    (e.g. "show me the labs" — for which patients? which lab?).
  - scope is clearly "comparison" but no comparison axis is hinted at.
  - A specific patient is referenced but no subject_id is given.

Prefer answering with the closest supported structured CompetencyQuestion over
asking a clarifying question when the user's intent is reasonably recoverable.
Clarifying questions are a last resort — they cost the clinician a round trip.

When you do set ``clarifying_question``, still populate
``interpretation_summary`` with what you interpreted so far (e.g. "unclear which
lab or cohort is meant") so the UI can explain why it's asking.
"""


_SELF_CHECK = """\
# Self-check Before Responding

Before you emit JSON, verify:

  1. Every ``clinical_concepts[].name`` is either a concrete concept or a
     category listed under "Concept Types".
  2. Every ``patient_filters[].field`` appears in "Supported Operations / filter",
     and the operator is allowed for that field.
  3. ``aggregation``, if set, appears in "Supported Operations / aggregate".
  4. If ``scope == "comparison"``, ``comparison_field`` is set and appears in
     "Supported Operations / comparison_axis".
  5. ``interpretation_summary`` is populated and restates the user's question
     in one clinician-readable sentence using the structured fields you chose.
  6. If any of (1)-(4) would otherwise force you to guess, set
     ``clarifying_question`` and leave the ambiguous field empty.
"""


# ---------------------------------------------------------------------------
# Operation-section rendering
# ---------------------------------------------------------------------------


def _operations_section(registry: OperationRegistry) -> str:
    """Render the Supported Operations section from the registry.

    Three sub-sections, one per kind. Each row is produced by the operation's
    ``describe_for_prompt`` method, so adding a new operation updates the
    prompt automatically. No drift possible — the prompt<->registry round-trip
    test enforces this.
    """
    lines: list[str] = ["# Supported Operations", ""]
    lines.append(
        "Use ONLY the names listed here for patient_filters[].field, "
        "aggregation, and comparison_field. Any other value triggers a retry."
    )
    for kind, label in [
        ("filter", "filter  (used in patient_filters[].field)"),
        ("aggregate", "aggregate  (used in aggregation)"),
        ("comparison_axis", "comparison_axis  (used in comparison_field)"),
    ]:
        lines.append("")
        lines.append(f"## {label}")
        lines.append("")
        lines.append(registry.describe_for_prompt(kind))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


PROMPT_EXAMPLES_DIR = Path(__file__).parent / "prompt_examples"


def _load_single_cq_examples() -> list[dict]:
    """Load every JSON file under ``prompt_examples/single_cq/``.

    Files are sorted by filename so numbering (``01_``, ``02_``, …) controls
    presentation order. Each file is a CompetencyQuestion JSON payload
    (Shape A — see Output Schema).
    """
    single_dir = PROMPT_EXAMPLES_DIR / "single_cq"
    if not single_dir.exists():
        return []
    return [
        json.loads(p.read_text())
        for p in sorted(single_dir.glob("*.json"))
    ]


def _load_big_question_examples() -> list[dict]:
    """Load every JSON file under ``prompt_examples/big_question/``.

    Each file is a Shape B payload (``narrative`` + ``competency_questions``).
    Phase 4.5: these teach the LLM to emit the multi-CQ shape for broad
    questions that share a single downstream knowledge graph.
    """
    big_dir = PROMPT_EXAMPLES_DIR / "big_question"
    if not big_dir.exists():
        return []
    return [
        json.loads(p.read_text())
        for p in sorted(big_dir.glob("*.json"))
    ]


def _examples_section() -> str:
    """Render the Examples section.

    Single-CQ examples are rendered first (Shape A), then big-question
    examples under a subheading (Shape B). Each example carries a question
    header so the LLM can match example structure to intent.
    """
    single = _load_single_cq_examples()
    big = _load_big_question_examples()
    if not single and not big:
        return "# Examples\n\n(no examples provided)\n"

    parts: list[str] = ["# Examples", ""]
    if single:
        parts.append("## Shape A — single CompetencyQuestion")
        parts.append("")
        for ex in single:
            question = ex.get("original_question", "<no question>")
            parts.append(f'Question: "{question}"')
            parts.append("```json")
            parts.append(json.dumps(ex, indent=2))
            parts.append("```")
            parts.append("")
    if big:
        parts.append("## Shape B — big-question decomposition")
        parts.append("")
        for ex in big:
            # big-question examples have no top-level original_question, so
            # we show the first sub-CQ's question plus the narrative for context.
            sub_qs = ex.get("competency_questions", [])
            first_q = sub_qs[0].get("original_question", "<no question>") if sub_qs else "<no question>"
            parts.append(f'Big question (example lead-in): "{first_q}"')
            parts.append("```json")
            parts.append(json.dumps(ex, indent=2))
            parts.append("```")
            parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


_PROMPT_HEADER = """\
You are a clinical question decomposer for an ICU analytics system built on MIMIC-IV data.
Your ONLY job is to translate a natural-language clinical question into a structured JSON object.
You must NEVER generate SQL, SPARQL, or compute any results — only structure the question.
"""


def build_system_prompt(registry: OperationRegistry) -> str:
    """Assemble the full decomposition system prompt.

    Sections in order:
        header → Role & Pipeline → Decomposition Goals → Ontology
        → Concept Types → Output Schema → Supported Operations
        → Return Type Inference → Scope Inference → Examples

    Every structural section lives in exactly one place: either a constant in
    this module or a JSON file under ``prompt_examples/``. Rebuilding from
    the registry ensures the "Supported Operations" section reflects the
    current set of filter / aggregate / comparison_axis operations.
    """
    return "\n".join([
        _PROMPT_HEADER,
        _ROLE_AND_PIPELINE,
        _DECOMPOSITION_GOALS,
        _ONTOLOGY,
        _CONCEPT_TYPES,
        _OUTPUT_SCHEMA,
        _operations_section(registry),
        _RETURN_TYPE_INFERENCE,
        _SCOPE_INFERENCE,
        _WHEN_TO_CLARIFY,
        _BIG_QUESTION,
        _SELF_CHECK,
        _examples_section(),
    ])


DECOMPOSITION_SYSTEM_PROMPT: str = build_system_prompt(get_default_registry())
"""The default-registry system prompt, built once at import time.

Call sites that want a custom registry should call ``build_system_prompt``
directly. Editing the prompt means editing either (a) one of the static
``_SECTION`` strings above, (b) an example JSON under ``prompt_examples/``,
or (c) an operation's ``describe_for_prompt`` in ``operations*.py`` — never
this module-level constant."""


# ---------------------------------------------------------------------------
# Answer-generation prompt (unchanged from Phase 0)
# ---------------------------------------------------------------------------


ANSWER_GENERATION_SYSTEM_PROMPT = """\
You are a clinical data analyst generating concise summaries from structured query results.

Rules:
- Write 1-3 sentences summarising the key findings.
- Cite specific numbers from the data (means, counts, ranges, percentages).
- NEVER hallucinate or infer values not present in the provided results.
- Use precise clinical terminology (e.g., "serum creatinine" not just "creatinine level").
- If the result set is empty, state clearly that no matching data was found.
- Do not include caveats about data quality unless the results themselves indicate issues.
"""


# ---------------------------------------------------------------------------
# Message builder (unchanged from Phase 0)
# ---------------------------------------------------------------------------


def build_decomposition_messages(
    question: str,
    conversation_history: list | None = None,
) -> list[dict]:
    """Build the messages array for the decomposition API call.

    Parameters
    ----------
    question:
        The new user question to decompose.
    conversation_history:
        Optional list of ``(CompetencyQuestion, AnswerResult)`` tuples from
        prior turns.  The last 5 turns are included to give the LLM context
        for follow-up questions.

    Returns
    -------
    list[dict]
        Messages suitable for ``client.messages.create(messages=...)``.
    """
    messages: list[dict] = []

    if conversation_history:
        for cq, _ in conversation_history[-5:]:
            messages.append({"role": "user", "content": cq.original_question})
            messages.append({"role": "assistant", "content": json.dumps(
                cq.model_dump(mode="json"), separators=(",", ":"),
            )})

    messages.append({"role": "user", "content": question})
    return messages
