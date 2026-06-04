# RDF Graph Building: Status Summary

_Drafted 2026-06-04. A code-grounded snapshot of how the RDF knowledge graph
is built, how questions get routed to it, where the construction algorithm
lives, and how the graph can be queried. Supersedes the build-focused parts of
`docs/2026-05-25/graph-db-causal-inference-status.md` (which remains the
reference for the causal/association **consumption** gap)._

---

## Status at a glance

| Capability | State | Primary location |
|---|---|---|
| Offline batch graph build (cohort → `.nt` file) | **WIRED** | `src/graph_construction/pipeline.py:49` (`build_graph`) |
| Per-question in-memory graph build (chat) | **WIRED** | `src/conversational/graph_builder.py:39` (`build_query_graph`) |
| Vannieuwenhuyze et al. (2025) algorithm (OWL-Time + Allen) | **IMPLEMENTED** | `src/graph_construction/event_writers.py`, `src/graph_construction/temporal/allen_relations.py` |
| Query routing (SQL-fast vs. graph) | **WIRED** | `src/conversational/planner.py:84`, `src/conversational/orchestrator.py:296-342` |
| SPARQL query engine (templated) | **WIRED** | `src/conversational/reasoner.py` |
| Offline query/export paths (NetworkX, PyG, Neo4j) | **PARTIAL** | `src/graph_analysis/`, `src/gnn/graph_export.py` |
| Comorbidity nodes | **DEAD CODE** | `src/graph_construction/event_writers.py:502` (`write_comorbidity`, never called) |
| Drug-category resolution at build time | **ABSENT** | drug names stored as free-text literals only |

The graph **builder is the mature part of the system**. The open work is
downstream — the causal/association layer that should *consume* the graph (see
the May 25 doc) — not the construction itself.

---

## 1. Where the construction algorithm lives

The construction code lives under **`src/graph_construction/`** and is shared by
two entry points that build the same RDF shape from different data sources.

### Research basis
The graph implements the conceptual model from **Vannieuwenhuyze, Mimouni & Du
Mouza (2025), "A Conceptual Model for Discovering Implicit Temporal Knowledge in
Clinical Data" (ER 2025)** — cited at `README.md:408` and `docs/architecture.md:5-7`.
Clinical events are modeled as OWL-Time instants or proper intervals, and the
six clinically-relevant Allen interval relations are computed pairwise within
each ICU stay.

### The two build pipelines

**(a) Offline / batch — `src/graph_construction/pipeline.py:49` `build_graph()`**
Builds a graph for a whole cohort from the DuckDB database and serializes it to
an NTriples file. The 7-step flow (documented at `pipeline.py:66-75`):
1. Connect to DuckDB; create derived tables (`age`, `readmission_labels`) — `pipeline.py:94-97, 257`.
2. Select the cohort by ICD-10 prefixes — `pipeline.py:103` (`select_neurology_cohort`).
3. Open a disk-backed graph and load the ontologies — `pipeline.py:113-120`.
4. Per patient, write demographics → admissions → ICU stays → ICU days → events → diagnoses — `pipeline.py:355` (`_process_patient`).
5. Compute Allen relations per patient — `pipeline.py:474-477`.
6. Optionally attach SNOMED-CT codes via `SnomedMapper` — `pipeline.py:123-137`.
7. Serialize to `.nt` — `pipeline.py:232-239`.
Parallelism: per-patient worker processes that each emit an NTriples fragment,
merged into the main graph — `pipeline.py:157-202`, `_process_single_patient` at `pipeline.py:316`.
CLI entry: `python -m src.graph_construction` (`src/graph_construction/__main__.py`),
also exposed as `make graph` (`Makefile:74`).

**(b) Per-question / in-memory — `src/conversational/graph_builder.py:39` `build_query_graph()`**
Materializes a small rdflib graph from a conversational `ExtractionResult`
(one cohort's worth of data), used once per chat turn and then discarded. It is
explicitly a **bridge, not new construction logic** (`graph_builder.py:1-5`) — it
calls the exact same writer functions as the batch pipeline (imports at
`graph_builder.py:19-36`). Invoked from the orchestrator at
`orchestrator.py:354`. Supports a parallel per-patient build path
(`graph_builder.py:313` `_build_parallel`).
> Note: the most recent graph change (`b53cc99`, 2026-06) added a per-patient
> `hadm→stay` index here (`graph_builder.py:116-126`) to kill an
> O(patients×stays) hang on cohort-wide questions.

### The shared writers (the algorithm itself)

- **Ontology / namespaces** — `src/graph_construction/ontology.py`. `initialize_graph()` (line 18) loads `ontology/definition/base_ontology.rdf` + `extended_ontology.rdf` and binds the `mimic:`, `time:`, `sct:` namespaces. The MIMIC namespace is `http://www.cnam.fr/MIMIC4-ICU-BSI/V1#` (`ontology.py:13`).
- **Patients & admissions** — `src/graph_construction/patient_writer.py`. `write_patient`, `write_admission`, `link_sequential_admissions`. The readmission outcome flags are attached here: `readmittedWithin30Days` / `readmittedWithin60Days` at `patient_writer.py:93-94`; `hasAdmission` at `patient_writer.py:103`; `followedBy` (sequential admissions) at `patient_writer.py:119`.
- **ICU structure & clinical events** — `src/graph_construction/event_writers.py`:
  - `write_icu_stay` (`:98`, a `time:ProperInterval`) and `write_icu_days` (`:141`, partitions a stay into per-day intervals).
  - Instant events: `write_biomarker_event` (`:212`), `write_clinical_sign_event` (vitals, `:276`), `write_microbiology_event` (`:332`).
  - Interval events: `write_prescription_event` (`:389`, a `time:ProperInterval` with `hasBeginning`/`hasEnd`).
  - Non-temporal: `write_diagnosis_event` (`:461`).
  - Containment edges written here: `containsICUStay` (`:136`), `hasICUDay`/`partOf` (`:197-198`), `associatedWithICUStay`/`hasICUStayEvent` (e.g. `:264-265`), `hasDiagnosis`/`diagnosisOf` (`:496-497`).
- **Allen temporal relations** — `src/graph_construction/temporal/allen_relations.py`. The six OWL-Time predicates are mapped at `allen_relations.py:15-22`; the pure pairwise classifier is `_classify_allen_relation` (`:25`); per-stay computation `compute_allen_relations` (`:169`) uses a batched-SPARQL bounds fetch plus before-chain pruning; per-patient driver `compute_allen_relations_for_patient` (`:230`).
- **Disk persistence** — `src/graph_construction/disk_graph.py`. Oxigraph/RocksDB-backed rdflib graph so cohort-scale builds don't OOM (`open_disk_graph` at `:26`).
- **SNOMED-CT enrichment** — `src/graph_construction/terminology/` (`SnomedMapper`), threaded through every writer's optional `snomed_mapper=` arg.

---

## 2. How queries get routed to the graph path

Routing happens once per sub-question inside the conversational orchestrator.

1. **Decompose** the natural-language question into one or more
   `CompetencyQuestion`s — `orchestrator.py:223` (`decompose_question`).
2. **Classify** each CQ with the planner — `orchestrator.py:297`
   (`self._planner.classify(cq)`). The planner is
   `src/conversational/planner.py`, `QueryPlanner.classify` at `planner.py:84`.
   It returns one of four `QueryPlan` values (`planner.py:43-65`):
   - `SQL_FAST` — single-concept aggregate / comparison / diagnosis-list /
     mortality; skips the graph entirely and runs one SQL query.
   - `GRAPH` — anything needing the RDF graph.
   - `CAUSAL` — `scope=="causal_effect"` with an intervention set of size ≥ 2
     (`planner.py:102-105`).
   - `SIMILARITY` — `scope=="patient_similarity"` (`planner.py:95-96`).
3. **Dispatch** on the plan — `orchestrator.py:298-342`. Only the `else`
   branch (`orchestrator.py:334-342`) is the **graph path**: it appends the CQ
   to `graph_cqs` and runs `_extract(...)`.

### What forces a CQ onto the graph path
From `QueryPlanner.classify` (`planner.py:109-151`), a CQ goes to `GRAPH` when:
- it has any `temporal_constraints` — Allen relations live only in the graph (`planner.py:110-111`);
- it has **no** clinical concepts (`planner.py:117-118`);
- it has more than one clinical concept (`planner.py:121-122`);
- its single concept is not a SQL-fast-eligible type — the eligible set is `{biomarker, vital, drug, diagnosis, microbiology, outcome}` at `planner.py:71-73`, checked at `planner.py:123-124`;
- its aggregation isn't SQL-compilable (no portable `sql_fn`, e.g. **median**) — `planner.py:138-140`;
- it's a raw-values (no aggregation) biomarker/vital query — `planner.py:128-136`;
- a comparison axis lacks a registered `sql_group_by` — `planner.py:144-149`.

Everything that does **not** trip one of those conditions takes the SQL
fast-path. The graph path is, by design, the fallthrough for "anything the
single-query SQL compiler can't faithfully express" (`planner.py:32`).

### Graph-path execution after routing
All graph-routed CQs in a turn share **one** graph (`orchestrator.py:344-358`):
the extractions are merged (`merge_extractions`), `build_query_graph` is called
once, and Allen relations are computed only if some CQ actually needs them
(`skip_allen_relations=not any_temporal`, `orchestrator.py:351-357`). Each CQ is
then answered by `reason(graph, cq)` (`orchestrator.py:360`).

---

## 3. How the graph can be queried

There are several query surfaces; the live chat path is SPARQL-via-`reasoner`.

### (a) SPARQL template engine — the production path
`src/conversational/reasoner.py`. Entry point `reason(graph, cq)` (`reasoner.py:564`):
- `select_templates(cq)` (`reasoner.py:450`) maps a CQ to one or more named
  templates.
- `build_sparql(...)` (`reasoner.py:514`) fills a template with the concept name
  and prepends the prefix block (`reasoner.py:38-43`).
- `graph.query(sparql)` is executed and rows are coerced to dicts
  (`reasoner.py:591`, `_result_to_dicts` at `:56`).

The template library `TEMPLATES` (`reasoner.py:82-404`) covers ~20 patterns,
including: `value_lookup` / `value_with_timestamps`, `aggregation_mean|max|min|count`,
`patient_list_by_diagnosis`, `drug_lookup`, `temporal_before` / `temporal_during`
(the Allen-relation queries), `icu_length_of_stay`, `admission_details`,
`patient_demographics`, `microbiology_results`, `comparison_two_groups` /
`comparison_by_field`, and `mortality_count`. Aggregate/comparison dispatch is
delegated to the `OperationRegistry` (`reasoner.py:473-489`), and aggregate
post-processing (e.g. median in Python) plugs in at `reasoner.py:604-612`.
The results feed `generate_answer(...)` for the chat UI (`orchestrator.py:362`).

### (b) Raw SPARQL via rdflib
Any holder of the `rdflib.Graph` can call `graph.query(...)` directly. The
codebase already does this internally — e.g. the Allen computer's batched bounds
query (`allen_relations.py:132-153`) and the feature extractor's `ASK` probe for
temporal predicates (`src/feature_extraction/graph_features.py:40-49`). The
prefix block to reuse is in `reasoner.py:38-43` (or bind via
`disk_graph.bind_namespaces`, `disk_graph.py:85`).

### (c) Graph-structure analysis (NetworkX)
`src/graph_analysis/rdf_to_networkx.py:28` `rdf_to_networkx()` converts the RDF
graph into a `networkx.DiGraph` for topology metrics. Driven by
`src/graph_analysis/analysis.py` and `python -m src.graph_analysis` / `make analyze`
(`Makefile:77`). This is how Layer-3 graph metrics and Layer-4 graph features
(`src/feature_extraction/graph_features.py`) read the graph.

### (d) GNN export (PyTorch Geometric)
`src/gnn/graph_export.py:1-33` converts the full RDF graph into a PyG
`HeteroData` object (9 node types) for GNN training — `make export-graph`
(`Makefile:91`).

### (e) Neo4j / Cypher — infrastructure, not yet wired to chat
`src/graph_analysis/neo4j_import.py` provides `import_rdf_to_neo4j` (`:36`, via
the n10s plugin) and `check_neo4j_connection` (`:11`). It is standalone
infrastructure — nothing in the conversational pipeline routes to Cypher today.
The May 25 design note recommends keeping the in-process rdflib/SPARQL path as
the substrate and deferring Neo4j until there's a concrete scale reason
(`docs/2026-05-25/graph-db-causal-inference-status.md:203-215`).

### (f) Persisted artifact
The batch build writes `data/processed/knowledge_graph.nt` (NTriples;
`pipeline.py:232-239`, `__main__.py:38`) and an Oxigraph store under
`oxigraph_store/` (`pipeline.py:114`). Either can be reloaded and queried with
any of the surfaces above.

---

## 4. Known gaps in the build (carried from the May 25 review, re-verified)

These are construction-side caveats, still accurate as of this snapshot:

- **Comorbidities are dead code.** `write_comorbidity` exists
  (`event_writers.py:502`) but is **not imported or called** by either
  `graph_builder.py` (imports at `:19-36`) or `pipeline.py` (imports at
  `:28-36`). No `Comorbidity` nodes ever enter the graph. Wiring it is one
  import + one call site in each pipeline.
- **No build-time drug categorization.** Drug names are stored as free-text
  literals on `PrescriptionEvent.hasDrugName` (`event_writers.py:432`); there is
  no RxNorm/ATC class mapping, so resolving "antibiotic" → ingredient set is not
  available at query time.
- **Cohort filtering is single-concept.** The conversational extractor filters
  by individual clinical concepts, not multi-event predicates (e.g. "antibiotic
  within 24h of admission"); such windows are expressible only as SPARQL
  post-filters on the materialized graph — see
  `docs/2026-05-25/graph-db-causal-inference-status.md:88-100`.

---

## Critical files (read these first)

- `src/graph_construction/pipeline.py:49` — `build_graph` (offline batch pipeline).
- `src/conversational/graph_builder.py:39` — `build_query_graph` (per-question pipeline; the seam shared with batch).
- `src/graph_construction/event_writers.py:98-499` — ICU/event node + edge writers; `:502` is the dead `write_comorbidity`.
- `src/graph_construction/patient_writer.py:93-119` — readmission outcome flags + `hasAdmission`/`followedBy`.
- `src/graph_construction/temporal/allen_relations.py:15-22, 25, 169, 230` — Allen relation predicates and computation.
- `src/graph_construction/ontology.py:18` — ontology loading / namespaces.
- `src/conversational/planner.py:84-151` — query routing (`QueryPlan`, `classify`).
- `src/conversational/orchestrator.py:296-369` — chat dispatch → extract → `build_query_graph` → `reason`.
- `src/conversational/reasoner.py:82-404, 450, 564` — SPARQL template library + `reason()` entry point.
- `src/graph_analysis/rdf_to_networkx.py:28`, `src/graph_analysis/neo4j_import.py:36`, `src/gnn/graph_export.py` — alternate query/export surfaces.
- `README.md:408`, `docs/architecture.md:5-7`, `docs/ontology.md` — algorithm references.
