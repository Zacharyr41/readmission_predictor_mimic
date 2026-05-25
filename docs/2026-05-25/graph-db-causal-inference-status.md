# Graph-DB-Backed Causal Inference: Current State

_Status report drafted 2026-05-25. Snapshot of `main` at the time of writing._

## TL;DR

**There is no graph-DB-backed causal inference path in this repo today.** Neo4j is wired up as an *optional, isolated module* (`src/graph_analysis/neo4j_import.py`) that is never called by any code outside its own tests. The causal layer (`src/causal/`) operates exclusively on DuckDB/BigQuery via SQL. The conversational router has a `QueryPlan.GRAPH` plan, but it points at **SPARQL over the in-process RDF knowledge graph**, not at Neo4j. There is no DAG discovery, no backdoor-criterion code, no adjustment-set derivation from graph structure anywhere in `src/causal/`.

---

## 1. Three "graph" things to disambiguate

The codebase has three graph representations; only one is a true graph database, and it's the unused one:

| Representation | Lives in | Used by | Touches Neo4j? |
|---|---|---|---|
| **RDF knowledge graph** (oxigraph disk-backed) | `src/graph_construction/pipeline.py` | SPARQL queries (`src/conversational/reasoner.py`), GNN export | No |
| **PyG HeteroData** (in-memory tensors) | `src/gnn/graph_export.py` | GNN training (`src/gnn/`) | No |
| **Neo4j graph database** | `src/graph_analysis/neo4j_import.py` | Tests only | Yes, but unused outside tests |

The Neo4j layer is the only true "graph DB" in the architecture sense. Everything else is either an in-memory file format (RDF .nt) or PyTorch tensors.

---

## 2. Neo4j stack — defined, isolated, unused

**Driver / connection code** — `src/graph_analysis/neo4j_import.py`:

- `check_neo4j_connection(uri, user, password)` at lines 11–34
- `import_rdf_to_neo4j(...)` at lines 36–111 (uses the n10s/neosemantics plugin via `CALL n10s.rdf.import.fetch(...)`)
- `query_neo4j(cypher, ...)` at lines 114–144
- `clear_neo4j_database(...)` at lines 146–163

**Configuration** — `config/settings.py:20-23`:

```python
neo4j_uri: str = Field(default="bolt://localhost:7687")
neo4j_user: str = Field(default="neo4j")
neo4j_password: str = Field(default="password")
```

**Re-exports** — `src/graph_analysis/__init__.py:19-40` re-exports those four functions.

**Who imports them?** A repo-wide grep for `import_rdf_to_neo4j`, `query_neo4j`, `check_neo4j_connection`, `clear_neo4j_database`, `from neo4j`, and `neo4j.GraphDatabase` returns hits **only inside `src/graph_analysis/`** (the module itself plus its `__init__.py`). No other src/ file imports them. The pipeline orchestrator, the conversational layer, and the causal layer never call any of them.

**Docs explicitly call it optional**:

- `docs/troubleshooting.md:65` — "Neo4j is optional; the pipeline works without it."
- `README.md:65` — "**Optional**: Neo4j 5.x for graph visualization"
- `docs/wlst_pipeline_plan.md:388` — "Export to Neo4j for visual inspection (optional)"

Neo4j is positioned as a *visualization aid*, not as a data store the runtime queries.

---

## 3. The causal layer is SQL-only

Module contents — `src/causal/`:

```
__init__.py  _rxnav.py  cohort.py  covariates.py  estimators/
interventions.py  models.py  outcomes.py  run.py  treatment_assignment.py
```

**Backend contract is SQL-shaped**, not graph-shaped. A grep across `src/causal/` for `backdoor`, `adjustment.set`, `do.calculus`, `d.separat` returns **zero matches**. The single `confound` hit is a forward-looking comment in `src/causal/estimators/base.py:333` ("8h adds unmeasured-confounder sensitivity") — a future phase, not implemented.

**No Cypher anywhere in causal code.** A repo-wide grep for `Cypher`, `cypher`, `MATCH (` returns hits only in `src/graph_analysis/neo4j_import.py` itself.

This is consistent with the locked Phase 8 plan (`memory/project_phase8_causal_spec.md`): econml backbone, Neyman–Rubin framework, no graph-based DAG inference promised.

---

## 4. Router: `QueryPlan.GRAPH` ≠ graph DB

`src/conversational/planner.py:43-65` defines four plans:

```
SQL_FAST = "sql_fast"   # single SQL query
GRAPH = "graph"         # RDF knowledge graph (Allen relations, median, time-series…)
CAUSAL = "causal"       # → src.causal.run_causal
SIMILARITY = "similarity"
```

The module docstring at `planner.py:1-33` makes the routing intent explicit — "extract→build-graph→SPARQL sequence", and `QueryPlan.GRAPH`'s docstring at lines 49–51 says it "Requires the RDF knowledge graph". The plan dispatches to **SPARQL on rdflib**, implemented in `src/conversational/reasoner.py:1-3` ("SPARQL reasoning engine ... against an RDF knowledge graph"), not to Neo4j.

A causal CQ with `scope="causal_effect"` and ≥2 interventions takes `QueryPlan.CAUSAL` (`planner.py:102-105`), which routes to `src.causal.run_causal` — and per §3 above, that path never touches a graph DB.

---

## 5. Tests for the Neo4j layer

- `tests/test_graph_analysis/test_analysis.py:227-279` — class `TestNeo4jIntegration` with an autoskip fixture that bails when Neo4j isn't reachable.
- Coverage is **basic only**: an RDF import round-trip and a `MATCH (n:HospitalAdmission) RETURN count(n)` Cypher query.
- No causal-inference integration tests against the graph DB exist (because no such code path exists to test).
- Consistent with `memory/MEMORY.md`: "642+ tests; 3 skipped (neo4j)".

---

## 6. What's actually present vs absent

| Feature | Present? | Citation |
|---|---|---|
| Neo4j driver wrapper | Yes | `src/graph_analysis/neo4j_import.py:11-163` |
| Settings for Neo4j URI/user/pass | Yes | `config/settings.py:20-23` |
| RDF→Neo4j import helper (n10s) | Yes, callable | `src/graph_analysis/neo4j_import.py:36-111` |
| Cypher query helper | Yes, callable | `src/graph_analysis/neo4j_import.py:114-144` |
| Production caller of any of the above | **No** | grep result above |
| Cypher emitted from any user-facing route | **No** | no `Cypher`/`MATCH (` outside `neo4j_import.py` |
| Causal route consuming Neo4j | **No** | `src/causal/` contains no neo4j imports |
| DAG / backdoor / adjustment-set logic | **No** | grep returned zero in `src/causal/` |
| Confounder discovery from graph structure | **No** | confounders come from `CompetencyQuestion.covariate_spec` configuration |
| Mediator / instrument discovery from graph | **No** | not implemented |
| Dashboard UI surface for graph-DB causal | **No** | router has no `CAUSAL_GRAPH` plan |
| Neo4j containerization / live DB in CI | **No** | tests autoskip when DB unavailable |
| Roadmap doc for graph-DB causal | **No** | none found in `docs/`; `memory/project_critic_external_grounding.md` mentions PubMed grounding, not graph-DB causal |

---

## 7. The gap to "user asks a causal question and gets a graph-DB-backed answer"

Concretely, none of these pieces exist yet:

1. **A graph-DB-aware causal plan.** `src/conversational/planner.py` would need a new `QueryPlan.CAUSAL_GRAPH` (or a flag on `CAUSAL`) and the corresponding branch in `classify()`.
2. **Cypher generation for confounder/mediator discovery.** Something like *"given exposure node E and outcome node O, return common ancestors / blocking sets"*. No such builder exists.
3. **DAG-aware adjustment-set computation.** Either implemented natively (backdoor criterion, do-calculus) or by delegating to a library like `dowhy` / `pgmpy`. `src/causal/estimators/` only contains S/T/X-learner metalearners (`metalearners.py`) and propensity helpers (`_propensity.py`).
4. **A pipeline step that actually loads MIMIC into Neo4j.** `src/graph_construction/pipeline.py` writes RDF to disk and stops there; `import_rdf_to_neo4j` is never invoked from the orchestrator (`src/main.py`) or from anywhere else in `src/`.
5. **A backend adapter for the causal layer that can query Neo4j.** `src/causal/cohort.py` expects a `.execute(sql, params)` protocol; a graph-backed cohort would need a parallel `.execute_cypher(...)` interface.
6. **CI infrastructure for Neo4j** (containerized service or mocked Cypher), so the path can actually be tested.
7. **UI surface** in the dashboard showing the derived DAG / adjustment set to the user.

What works *adjacent* to this and could be reused: the RDF knowledge graph (`src/graph_construction/pipeline.py`) already encodes Patient / HospitalAdmission / ICUStay / event-node structure and Allen temporal relations — that's the substrate a future DAG-discovery layer would walk. The Neo4j import + Cypher helpers are also ready, so the missing piece is genuinely the *causal* logic on top, plus the routing.

---

## Bottom line

Neo4j is plumbing without a tenant. The causal-inference system answers questions the same way it answers aggregate SQL questions — by issuing SQL against DuckDB/BigQuery, with confounders supplied by the `CompetencyQuestion`'s configuration rather than discovered from the graph. If you want "graph-DB-backed causal inference" as a real product capability, the starting point is the routing layer downward — none of the load-bearing pieces (DAG discovery, adjustment-set computation, Cypher generation, Neo4j loading at pipeline runtime, `CAUSAL_GRAPH` plan, UI surface) exist today.
