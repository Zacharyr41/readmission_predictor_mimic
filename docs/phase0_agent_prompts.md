# Phase 0 Implementation — Agent Prompts

Each prompt below is designed to be given to a Claude Code agent to implement one portion of the Phase 0 conversational temporal analytics layer. Execute in order — each builds on the previous.

---

## Prompt 1: Foundation — Data Models & Configuration

```
You are implementing Phase 0 of a conversational temporal analytics tool for clinical EHR data. This prompt covers the foundation layer only.

CONTEXT:
- The project is a clinical ML pipeline at the repo root. It already has data ingestion (DuckDB), RDF/OWL graph construction, temporal reasoning (Allen algebra), and SPARQL querying.
- We are adding a new `src/conversational/` package that lets clinicians ask questions in plain English and get deterministic answers backed by on-the-fly knowledge graphs.
- Read the full plan at `docs/phase0_agent_prompts.md` and the architecture plan at `.claude/plans/synthetic-snuggling-hennessy.md` for full context.

TASK — Create these files:

1. `src/conversational/__init__.py` — Empty init file.

2. `src/conversational/models.py` — Pydantic data models that define the contracts between all pipeline stages. Include:
   - `ClinicalConcept(BaseModel)`: name (str), concept_type (Literal["biomarker", "vital", "drug", "diagnosis", "microbiology"]), attributes (list[str])
   - `TemporalConstraint(BaseModel)`: relation (str — "before", "after", "during", "within"), reference_event (str), time_window (str | None)
   - `PatientFilter(BaseModel)`: field (str), operator (str — ">", "<", "=", ">=", "<=", "contains", "in"), value (str)
   - `ReturnType(str, Enum)`: TEXT, TABLE, TEXT_AND_TABLE, VISUALIZATION
   - `CompetencyQuestion(BaseModel)`: original_question (str), clinical_concepts (list[ClinicalConcept]), temporal_constraints (list[TemporalConstraint]), patient_filters (list[PatientFilter]), aggregation (str | None), return_type (ReturnType), scope (Literal["single_patient", "cohort", "comparison"])
   - `ExtractionResult(BaseModel)`: patients (DataFrame-like list[dict]), admissions (list[dict]), icu_stays (list[dict]), events (dict[str, list[dict]] keyed by concept_type)
   - `AnswerResult(BaseModel)`: text_summary (str), data_table (list[dict] | None), table_columns (list[str] | None), visualization_spec (dict | None), graph_stats (dict), sparql_queries_used (list[str])

3. Update `config/settings.py` — Add an optional `anthropic_api_key: str | None = None` field to the existing Settings class. Read from env var `ANTHROPIC_API_KEY`.

4. Update `pyproject.toml` — Add a `conversational` extras group: `anthropic>=0.39`, `streamlit>=1.40`, `plotly>=5.0`.

5. `tests/test_conversational/__init__.py` — Empty init.

6. `tests/test_conversational/test_models.py` — Unit tests for all Pydantic models: construction, serialization to/from JSON, validation of enum fields, rejection of invalid concept_types.

Run the tests after writing them to confirm they pass.
```

---

## Prompt 2: LLM Prompt Templates & Decomposer

```
You are implementing the LLM translation layer for Phase 0 of a conversational temporal analytics tool.

CONTEXT:
- Read `.claude/plans/synthetic-snuggling-hennessy.md` for the full architecture.
- `src/conversational/models.py` already exists with CompetencyQuestion, ClinicalConcept, TemporalConstraint, PatientFilter, ReturnType.
- The OWL ontology schema is documented in `docs/ontology.md` — read it thoroughly. The class hierarchy (Patient, HospitalAdmission, ICUStay, ICUDay, BioMarkerEvent, ClinicalSignEvent, MicrobiologyEvent, PrescriptionEvent, DiagnosisEvent) and all properties are defined there.
- The LLM has ONE job: translate English ↔ structured format. It must NEVER generate SQL, SPARQL, or compute anything.

TASK — Create these files:

1. `src/conversational/prompts.py` — Two prompt templates:

   a. `DECOMPOSITION_SYSTEM_PROMPT`: A system prompt for Claude that:
      - Describes the role: "You decompose clinical questions into structured components"
      - Includes the ontology class hierarchy and available concept types (biomarker, vital, drug, diagnosis, microbiology)
      - Lists example biomarker names (creatinine, lactate, sodium, glucose, INR, troponin, hemoglobin, etc.)
      - Lists example vital names (heart rate, blood pressure, respiratory rate, SpO2, temperature, MAP)
      - Lists example drug categories (vasopressors, antibiotics, sedatives, anticoagulants)
      - Instructs Claude to output ONLY valid JSON matching the CompetencyQuestion schema
      - Includes 6-8 few-shot examples covering: single-value lookup, cohort aggregation, temporal query, diagnosis search, drug lookup, comparison, trend, and visualization request
      - Explains how to infer ReturnType: default TEXT_AND_TABLE, VISUALIZATION only for explicit "plot"/"chart"/"graph"/"visualize"/"distribution" keywords
      - Explains scope inference: "single_patient" when a specific patient is mentioned, "cohort" for population questions, "comparison" for group-vs-group

   b. `ANSWER_GENERATION_SYSTEM_PROMPT`: A system prompt for Claude that:
      - Takes the original question and structured query results (as JSON rows)
      - Generates a 1-3 sentence clinical summary citing specific numbers from the data
      - Never hallucinate values — only reference data present in the results
      - Be precise with clinical terminology

   c. A helper function `build_decomposition_messages(question: str, conversation_history: list | None) -> list[dict]` that constructs the messages array for the API call, including the system prompt and optionally prior CompetencyQuestion/AnswerResult pairs for follow-up context.

2. `src/conversational/decomposer.py` — The decomposition module:
   - Function `decompose(client: anthropic.Anthropic, question: str, conversation_history: list | None = None) -> CompetencyQuestion`
   - Calls `client.messages.create()` with model="claude-sonnet-4-20250514", the decomposition system prompt, and the user question
   - Extracts JSON from Claude's response (handle both raw JSON and markdown code blocks)
   - Parses into CompetencyQuestion via Pydantic (model_validate)
   - On validation failure: retry once with the error message appended
   - Post-processing: if no explicit visualization keywords found in original_question, force return_type to TEXT_AND_TABLE

3. `tests/test_conversational/test_decomposer.py` — Tests that mock `anthropic.Anthropic`:
   - Test successful decomposition with canned Claude response
   - Test follow-up question receives conversation history
   - Test JSON extraction from markdown code block
   - Test validation retry on malformed JSON
   - Test return_type defaults to TEXT_AND_TABLE

Run all tests after writing.
```

---

## Prompt 3: Question-Driven Data Extraction

```
You are implementing the data extraction layer for Phase 0 of a conversational temporal analytics tool.

CONTEXT:
- Read `.claude/plans/synthetic-snuggling-hennessy.md` for the full architecture.
- `src/conversational/models.py` has CompetencyQuestion, ClinicalConcept, PatientFilter, TemporalConstraint, ExtractionResult.
- The MIMIC-IV data lives in a DuckDB database. The existing extraction queries in `src/graph_construction/pipeline.py` (read the _query_* functions at lines 482-851) show the table schemas: labevents JOIN d_labitems, chartevents JOIN d_items, prescriptions, diagnoses_icd JOIN d_icd_diagnoses, microbiologyevents, patients, admissions, icustays.
- This module generates SQL from the structured CompetencyQuestion — it does NOT use an LLM.

TASK — Create:

1. `src/conversational/extractor.py` — Template-based SQL generation and DuckDB extraction:

   - Function `extract(db_path: Path, cq: CompetencyQuestion) -> ExtractionResult`
   - Opens a read-only DuckDB connection

   Core logic:
   a. Build patient/admission base query from PatientFilters:
      - field="age" → JOIN patients p JOIN age a ON p.subject_id = a.subject_id, WHERE a.age {operator} {value}
      - field="gender" → WHERE p.gender = '{value}'
      - field="diagnosis" → JOIN diagnoses_icd di, WHERE di.icd_code LIKE '{value}%' (ICD prefix) or JOIN d_icd_diagnoses WHERE long_title ILIKE '%{value}%'
      - field="admission_type" → WHERE a.admission_type = '{value}'
      - No filters → include all patients/admissions

   b. For each ClinicalConcept, generate targeted SQL:
      - concept_type="biomarker": SELECT from labevents l JOIN d_labitems d ON l.itemid = d.itemid WHERE d.label ILIKE '%{name}%' AND l.valuenum IS NOT NULL, filtered to the relevant hadm_ids and time windows
      - concept_type="vital": SELECT from chartevents c JOIN d_items d ON c.itemid = d.itemid WHERE d.label ILIKE '%{name}%' AND c.valuenum IS NOT NULL, filtered to relevant stay_ids
      - concept_type="drug": SELECT from prescriptions WHERE drug ILIKE '%{name}%', filtered to relevant hadm_ids
      - concept_type="diagnosis": SELECT from diagnoses_icd di JOIN d_icd_diagnoses d WHERE d.long_title ILIKE '%{name}%' OR di.icd_code LIKE '{name}%', filtered to relevant hadm_ids
      - concept_type="microbiology": SELECT from microbiologyevents WHERE spec_type_desc ILIKE '%{name}%' OR org_name ILIKE '%{name}%', filtered to relevant hadm_ids and time windows

   c. Apply TemporalConstraints as time-window filters:
      - "within" + time_window → charttime BETWEEN intime AND intime + INTERVAL '{time_window}'
      - "during" + reference_event="ICU stay" → charttime BETWEEN intime AND outtime
      - "before"/"after" reference events mapped to appropriate timestamp comparisons

   d. Also extract the corresponding patients, admissions, and icustays rows for the matched data (needed by graph_builder)

   e. Return ExtractionResult with all DataFrames as list[dict]

   IMPORTANT: Use parameterized queries (?) for integer/date values. Use ILIKE for name matching. Never concatenate user-provided values directly into SQL — the concept names come from the LLM decomposition which is controlled, but still follow safe practices.

2. `tests/test_conversational/test_extractor.py` — Tests using a synthetic DuckDB:
   - Check `tests/conftest.py` for the existing `synthetic_duckdb_with_events` fixture — reuse it if it has the needed tables (patients, admissions, icustays, labevents, chartevents, prescriptions, diagnoses_icd, d_labitems, d_items, d_icd_diagnoses, microbiologyevents). If it doesn't have all needed tables, create a local fixture in this test file.
   - Test biomarker extraction: build a CompetencyQuestion for "creatinine" and verify the extractor returns matching lab rows
   - Test patient filter: filter by age > 50 and verify only matching patients returned
   - Test temporal constraint: "within 48 hours of ICU admission" filters correctly
   - Test empty result: query for a nonexistent concept returns empty lists
   - Test multiple concepts: CompetencyQuestion with both a biomarker and a vital

Run all tests after writing.
```

---

## Prompt 4: Per-Query Graph Construction

```
You are implementing the per-query graph builder for Phase 0 of a conversational temporal analytics tool.

CONTEXT:
- Read `.claude/plans/synthetic-snuggling-hennessy.md` for the full architecture.
- `src/conversational/models.py` has ExtractionResult.
- The existing graph construction code is in `src/graph_construction/`:
  - `ontology.py`: `initialize_graph(ontology_dir)` loads the OWL ontology into an rdflib Graph, `MIMIC_NS`, `TIME_NS`
  - `patient_writer.py`: `write_patient(graph, patient_data)`, `write_admission(graph, adm_data, patient_uri)`, `link_sequential_admissions(graph, admission_uris)`
  - `event_writers.py`: `write_icu_stay(graph, stay_data, admission_uri)`, `write_icu_days(graph, stay_data, icu_stay_uri)`, `write_biomarker_event(graph, lab_data, icu_stay_uri, icu_day_metadata)`, `write_clinical_sign_event(graph, vital_data, icu_stay_uri, icu_day_metadata)`, `write_microbiology_event(graph, micro_data, icu_stay_uri, icu_day_metadata)`, `write_prescription_event(graph, rx_data, icu_stay_uri, icu_day_metadata)`, `write_diagnosis_event(graph, dx_data, admission_uri)`
  - `temporal/allen_relations.py`: `compute_allen_relations_for_patient(graph, patient_uri)`
- Read these files to understand the exact dict key formats expected by each writer function.

TASK — Create:

1. `src/conversational/graph_builder.py`:

   - Function `build_query_graph(ontology_dir: Path, extraction: ExtractionResult) -> tuple[Graph, dict]`
   - Returns (rdf_graph, stats_dict)

   Implementation:
   a. Create an in-memory rdflib Graph() — NOT disk-backed Oxigraph (per-query graphs are small)
   b. Load ontology: `initialize_graph(ontology_dir)` — copy all triples into the new graph, or parse the ontology files directly
   c. Group extracted data by patient (subject_id):
      - For each patient: call `write_patient(graph, patient_data_dict)`
      - For each admission of that patient: call `write_admission(graph, adm_data, patient_uri)`
      - For each ICU stay of that admission: call `write_icu_stay(graph, stay_data, admission_uri)` then `write_icu_days(graph, stay_data, icu_stay_uri)`
      - For each event: call the appropriate write_*_event function, passing the icu_stay_uri and icu_day_metadata
      - For diagnoses: call `write_diagnosis_event(graph, dx_data, admission_uri)`
   d. Call `link_sequential_admissions(graph, admission_uris)` if a patient has multiple admissions
   e. Call `compute_allen_relations_for_patient(graph, patient_uri)` for each patient
   f. Collect stats: total triples (len(graph)), patient count, event count by type
   g. Return (graph, stats)

   KEY: The dict formats passed to write_* functions must exactly match what those functions expect. Read the function signatures in event_writers.py and patient_writer.py carefully. The ExtractionResult stores events as list[dict] — map them to the expected key formats.

2. `tests/test_conversational/test_graph_builder.py`:
   - Create a synthetic ExtractionResult with 1 patient, 1 admission, 1 ICU stay, 2 lab events, 1 vital event, 1 prescription
   - Call build_query_graph() and verify:
     - Graph has > 0 triples
     - Patient node exists (SPARQL: SELECT ?p WHERE { ?p rdf:type mimic:Patient })
     - BioMarkerEvent nodes exist
     - Allen temporal relations were computed (at least some time:before triples exist)
     - Stats dict has correct counts
   - The ontology_dir should point to `ontology/definition/` in the repo

Run all tests after writing.
```

---

## Prompt 5: SPARQL Reasoning Engine

```
You are implementing the SPARQL reasoning engine for Phase 0 of a conversational temporal analytics tool.

CONTEXT:
- Read `.claude/plans/synthetic-snuggling-hennessy.md` for the full architecture.
- `src/conversational/models.py` has CompetencyQuestion with clinical_concepts, temporal_constraints, aggregation, scope.
- `docs/ontology.md` contains 10 example SPARQL queries (lines 241-421) — read them all. They demonstrate the exact SPARQL patterns that work against the ontology.
- The ontology namespaces: mimic = http://www.cnam.fr/MIMIC4-ICU-BSI/V1#, time = http://www.w3.org/2006/time#
- Key RDF patterns:
  - Patient → hasAdmission → HospitalAdmission → containsICUStay → ICUStay → hasICUDay → ICUDay
  - Event → associatedWithICUStay → ICUStay (and reverse: hasICUStayEvent)
  - BioMarkerEvent: hasBiomarkerType, hasValue, hasUnit, hasRefRangeLower/Upper
  - ClinicalSignEvent: hasClinicalSignName, hasValue
  - PrescriptionEvent: hasDrugName, hasDoseValue, hasDoseUnit, hasRoute (time:ProperInterval with hasBeginning/hasEnd)
  - DiagnosisEvent: hasIcdCode, hasLongTitle, hasSequenceNumber (linked via hasDiagnosis from admission)
  - Allen relations: time:before, time:inside, time:intervalMeets, time:intervalOverlaps, time:intervalStarts, time:intervalFinishes

TASK — Create:

1. `src/conversational/reasoner.py`:

   - A SPARQL template library as a dict mapping template names to parameterized SPARQL strings
   - Templates needed (~15-20):
     a. `value_lookup`: Get specific values for a concept type (biomarker/vital) — parameterized by concept name
     b. `value_with_timestamps`: Same but includes timestamps, ordered chronologically
     c. `aggregation_mean/max/min/count`: Aggregate values — parameterized by concept name and aggregation function
     d. `patient_list_by_diagnosis`: Find patients with a specific diagnosis (by ICD prefix or title text)
     e. `drug_lookup`: Get prescriptions for a patient/cohort — with start/end times
     f. `event_count_by_type`: Count events grouped by type
     g. `temporal_before`: Find events that occurred before a reference event (using time:before)
     h. `temporal_during`: Find events during an interval (using time:inside)
     i. `icu_length_of_stay`: Get ICU LOS from duration triples
     j. `admission_details`: Get admission type, discharge location, readmission flags
     k. `trend_over_time`: Get ordered values over time for a concept (for trend analysis)
     l. `comparison_two_groups`: Compare a metric between two filtered groups
     m. `patient_demographics`: Get age, gender for filtered patients
     n. `microbiology_results`: Get culture results
     o. `all_events_for_stay`: Get all events for an ICU stay, ordered by time

   - Function `select_templates(cq: CompetencyQuestion) -> list[str]`:
     Maps the CompetencyQuestion's concept_types + aggregation + scope to the appropriate template names.
     Logic:
     - concept_type="biomarker" + aggregation="mean" → "aggregation_mean"
     - concept_type="biomarker" + aggregation=None → "value_with_timestamps"
     - concept_type="diagnosis" + scope="cohort" → "patient_list_by_diagnosis"
     - concept_type="drug" → "drug_lookup"
     - temporal_constraint with "before" → add "temporal_before"
     - scope="comparison" → "comparison_two_groups"
     - Always include "patient_demographics" as context

   - Function `build_sparql(template_name: str, cq: CompetencyQuestion) -> str`:
     Takes a template and fills in the parameters from the CompetencyQuestion (concept names, patient filters, etc.)

   - Function `reason(graph: Graph, cq: CompetencyQuestion) -> tuple[list[dict], list[str]]`:
     1. Select templates via select_templates(cq)
     2. Build SPARQL for each template via build_sparql(template, cq)
     3. Execute each via graph.query(sparql) (rdflib native)
     4. Convert results to list[dict] (each row as a dict with column names as keys)
     5. Merge results from multiple queries into a combined result set
     6. Return (results, sparql_queries_used)

2. `tests/test_conversational/test_reasoner.py`:
   - Build a small RDF graph in the test fixture using the existing event writers (similar to test_graph_builder), with known data (e.g., a patient with creatinine=1.2 and heart_rate=80)
   - Test value_lookup: query for "creatinine" returns the expected value
   - Test aggregation: query for mean of a biomarker returns correct value
   - Test template selection: verify select_templates returns expected templates for different CompetencyQuestion configurations
   - Test empty results: query for nonexistent concept returns empty list

Run all tests after writing.
```

---

## Prompt 6: Answer Generation & Return Type Handling

```
You are implementing the answer generation layer for Phase 0 of a conversational temporal analytics tool.

CONTEXT:
- Read `.claude/plans/synthetic-snuggling-hennessy.md` for the full architecture.
- `src/conversational/models.py` has CompetencyQuestion (with return_type: ReturnType), AnswerResult.
- `src/conversational/prompts.py` has ANSWER_GENERATION_SYSTEM_PROMPT.
- The answerer is the SECOND and FINAL place the LLM is used. It translates structured query results into English. It does NOT compute or generate data.

TASK — Create:

1. `src/conversational/answerer.py`:

   - Function `generate_answer(client: anthropic.Anthropic, cq: CompetencyQuestion, results: list[dict], graph_stats: dict, sparql_queries: list[str]) -> AnswerResult`:

   Implementation:
   a. Build a user message containing:
      - The original question
      - The structured results as formatted JSON (truncate to first 50 rows if large)
      - The CompetencyQuestion's aggregation and scope for context

   b. Call `client.messages.create()` with:
      - model="claude-sonnet-4-20250514"
      - system=ANSWER_GENERATION_SYSTEM_PROMPT
      - The user message
      - max_tokens=500 (summaries should be concise)

   c. Extract text_summary from Claude's response

   d. Format data_table:
      - If results are non-empty: include the results as list[dict]
      - Rename technical column names to human-readable (e.g., "hasBiomarkerType" → "Biomarker", "hasValue" → "Value", "inXSDDateTimeStamp" → "Timestamp")
      - Set table_columns to the ordered list of column names

   e. Handle visualization:
      - If cq.return_type == ReturnType.VISUALIZATION and results are non-empty:
        - Make a second Claude API call asking it to generate a Plotly JSON spec for the requested visualization type
        - The prompt should include the data columns and sample rows, and ask for a minimal plotly.graph_objects spec (chart type, x, y, color, title)
        - Parse the returned JSON into visualization_spec
      - Otherwise: visualization_spec = None

   f. Return AnswerResult(text_summary, data_table, table_columns, visualization_spec, graph_stats, sparql_queries)

   - Helper function `_rename_columns(results: list[dict]) -> tuple[list[dict], list[str]]`:
     Maps ontology property names to human-readable names. Return (renamed_results, column_names).
     Mapping: hasBiomarkerType→"Biomarker", hasClinicalSignName→"Vital Sign", hasValue→"Value", hasUnit→"Unit", inXSDDateTimeStamp→"Timestamp", hasDrugName→"Drug", hasIcdCode→"ICD Code", hasLongTitle→"Diagnosis", hasAdmissionId→"Admission ID", hasSubjectId→"Patient ID", hasStayId→"Stay ID", numericDuration→"Duration (days)", hasAge→"Age", hasGender→"Gender", etc.

2. `tests/test_conversational/test_answerer.py`:
   - Mock anthropic.Anthropic to return canned text summary
   - Test with TEXT_AND_TABLE return type: verify text_summary populated, data_table has renamed columns, no visualization_spec
   - Test with VISUALIZATION return type: mock a second Claude call returning Plotly JSON, verify visualization_spec is populated
   - Test with empty results: verify text_summary still generated (something like "No matching data found"), data_table is None
   - Test column renaming: verify _rename_columns maps correctly

Run all tests after writing.
```

---

## Prompt 7: Orchestrator & Conversation Management

```
You are implementing the orchestrator that wires together the full Phase 0 pipeline.

CONTEXT:
- Read `.claude/plans/synthetic-snuggling-hennessy.md` for the full architecture.
- All pipeline stages already exist:
  - `src/conversational/decomposer.py`: `decompose(client, question, history) -> CompetencyQuestion`
  - `src/conversational/extractor.py`: `extract(db_path, cq) -> ExtractionResult`
  - `src/conversational/graph_builder.py`: `build_query_graph(ontology_dir, extraction) -> (Graph, stats)`
  - `src/conversational/reasoner.py`: `reason(graph, cq) -> (results, sparql_queries)`
  - `src/conversational/answerer.py`: `generate_answer(client, cq, results, stats, sparql_queries) -> AnswerResult`
- `config/settings.py` has Settings with anthropic_api_key, data fields (db_path equiv).

TASK — Create:

1. `src/conversational/orchestrator.py`:

   - Class `ConversationalPipeline`:
     ```
     __init__(self, db_path: Path, ontology_dir: Path, api_key: str)
     ```
     - Stores db_path, ontology_dir
     - Creates `self.client = anthropic.Anthropic(api_key=api_key)`
     - `self.conversation_history: list[tuple[CompetencyQuestion, AnswerResult]] = []`
     - `self.max_history = 10` (sliding window to avoid unbounded context growth)

     ```
     ask(self, question: str) -> AnswerResult
     ```
     - Step 1: `cq = decompose(self.client, question, self.conversation_history)`
     - Step 2: `extraction = extract(self.db_path, cq)`
     - Step 3: `graph, stats = build_query_graph(self.ontology_dir, extraction)`
     - Step 4: `results, sparql_queries = reason(graph, cq)`
     - Step 5: `answer = generate_answer(self.client, cq, results, stats, sparql_queries)`
     - Step 6: Append (cq, answer) to conversation_history (trim to max_history)
     - Return answer
     - Wrap in try/except: on any stage failure, return an AnswerResult with text_summary explaining the error and empty data

     ```
     reset(self)
     ```
     - Clears conversation_history

   - Factory function `create_pipeline_from_settings() -> ConversationalPipeline`:
     - Loads Settings from config/settings.py
     - Determines db_path from settings (typically `data/processed/mimiciv.duckdb`)
     - ontology_dir = Path("ontology/definition")
     - api_key from settings.anthropic_api_key
     - Returns ConversationalPipeline instance

2. `tests/test_conversational/test_orchestrator.py`:
   - End-to-end test with ALL external calls mocked:
     - Mock decomposer to return a canned CompetencyQuestion
     - Mock extractor to return a canned ExtractionResult
     - Mock graph_builder to return a small Graph with known triples
     - Mock reasoner to return canned results
     - Mock answerer to return a canned AnswerResult
   - Test single question: verify all stages called in order, correct result returned
   - Test conversation follow-up: call ask() twice, verify second call passes conversation_history to decomposer
   - Test error handling: mock extractor to raise an exception, verify AnswerResult has error text_summary
   - Test reset(): verify conversation_history is cleared
   - Test max_history: call ask() 12 times, verify history length is 10

Run all tests after writing.
```

---

## Prompt 8: Streamlit Chat UI

```
You are implementing the Streamlit chat UI for Phase 0 of a conversational temporal analytics tool.

CONTEXT:
- Read `.claude/plans/synthetic-snuggling-hennessy.md` for the full architecture.
- `src/conversational/orchestrator.py` has `ConversationalPipeline` with `.ask(question) -> AnswerResult` and `create_pipeline_from_settings()`.
- `src/conversational/models.py` has `AnswerResult` with: text_summary (str), data_table (list[dict] | None), table_columns (list[str] | None), visualization_spec (dict | None), graph_stats (dict), sparql_queries_used (list[str]).
- `ReturnType` enum has TEXT, TABLE, TEXT_AND_TABLE, VISUALIZATION.

TASK — Create:

1. `src/conversational/app.py` — Streamlit chat application:

   - Page config: `st.set_page_config(page_title="NeuroGraph", layout="wide")`
   - Title: "NeuroGraph — Conversational Clinical Analytics"

   - Sidebar:
     - DuckDB path input (default: "data/processed/mimiciv.duckdb") with file existence check
     - Anthropic API key input (password field, defaults to env var)
     - "Connect" button that initializes the pipeline in session_state
     - Connection status indicator (green/red)
     - "New Conversation" button that calls pipeline.reset() and clears chat
     - Expandable "About" section explaining what the tool does

   - Main area:
     - Display conversation history from st.session_state.messages (list of {"role": "user"/"assistant", "content": ...})
     - For each assistant message, display:
       a. text_summary as markdown via st.chat_message("assistant")
       b. data_table via st.dataframe() if not None (inside the same chat message)
       c. visualization via st.plotly_chart(go.Figure(visualization_spec)) if not None
       d. Expandable "Query Details" section showing graph_stats and sparql_queries_used (for auditability)
     - st.chat_input("Ask a clinical question...") at the bottom
     - On submit:
       - Append user message to st.session_state.messages
       - Display user message via st.chat_message("user")
       - Show st.spinner("Analyzing...") while processing
       - Call pipeline.ask(question)
       - Append assistant response to st.session_state.messages
       - Rerun to display

   - Session state initialization:
     - st.session_state.messages = []
     - st.session_state.pipeline = None (created on "Connect")

   - Error handling:
     - If pipeline not connected, show warning "Please connect to a database first"
     - If API call fails, display error in chat as assistant message

2. Add a convenience script `scripts/run_chat.sh`:
   ```bash
   #!/usr/bin/env bash
   streamlit run src/conversational/app.py
   ```

3. Update `src/conversational/__init__.py` with a brief module docstring and public exports:
   ```python
   from src.conversational.orchestrator import ConversationalPipeline, create_pipeline_from_settings
   from src.conversational.models import CompetencyQuestion, AnswerResult
   ```

Do NOT write tests for the Streamlit app (Streamlit apps are tested manually). Instead, verify the app launches without import errors by running: `.venv/bin/python -c "from src.conversational.app import *"` (this should not fail).
```

---

## Execution Notes

- **Run in order**: Each prompt builds on the previous. Prompt 1 creates the foundation that all others depend on.
- **Each prompt is self-contained**: An agent should be able to execute it with only the repo context and the referenced files.
- **Tests are mandatory**: Every prompt (except #8) includes tests. The agent should run them before completing.
- **No over-engineering**: Phase 0 is a proof-of-concept. Templates over magic, simple over clever.
