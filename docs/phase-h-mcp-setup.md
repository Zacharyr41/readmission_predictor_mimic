# Phase H — MCP backend setup guide

How to configure each of the optional Phase-G MCP backends that the
externally-grounded critic (and the contextualization sub-agent) can
call. Tiered by setup difficulty, not by clinical importance — so the
"easy wins" come first.

The Phase-H critic gracefully degrades on missing backends: any tool
that's not configured returns `{"status": "unavailable", ...}`, the
agent loop continues, and the verdict still renders. So you can ship
with just the easy tier and add the harder backends later.

Cross-refs:
- `docs/phase-h-externally-grounded-critic.md` — what the critic does
  with these backends.
- `docs/phase-h-smoke-test.md` — manual verification once a backend is
  configured.
- `scripts/critic_evidence_smoke.py` — automated probe that reports
  which configured backends actually fire.

## What you start with (no setup needed)

Three of the 11 critic tools work fully offline against files in the
repo:

- **`pubmed_search`** — defaults to NCBI E-utilities (public API, no
  auth). Throttled at 3 req/s without a key.
- **`loinc_reference_range`** — reads
  `data/ontology_cache/loinc_reference_ranges.json`.
- **`mimic_distribution_lookup`** — reads
  `data/processed/lab_distributions.json`. The Tier-D catalog is
  cohort-stratified (one entry per `(itemid, cohort)`). When the
  requested cohort is in `data/mappings/clinical_cohorts.json` but
  not in the cached catalog, the tool falls back to on-the-fly
  compute against the session's MIMIC backend (DuckDB or BigQuery —
  same dispatch as the rest of the pipeline via `DATA_SOURCE`).
- **`mimic_itemid_search`** — Tier-D follow-up. Live SQL against
  `d_labitems` + `d_items` to map a free-text analyte name (e.g.
  `"procalcitonin"`, `"heart rate"`) to ranked MIMIC itemid
  candidates. Same `DATA_SOURCE` dispatch as
  `mimic_distribution_lookup` — no new env vars. The critic uses
  this to chain `search → distribution_lookup` when it doesn't
  already know the itemid; empty results signal "not in MIMIC" and
  the model pivots to PubMed.

If both JSONs are present and you have internet for NCBI, you're at
the floor of what Phase H needs. The smoke test passes its "at least
one backend invoked" gate at this point.

### Regenerating the Tier-D catalog

The catalog is committed to the repo, but if you've added a new
cohort to `clinical_cohorts.json` or want to refresh against a
newer MIMIC snapshot, regenerate with:

```bash
.venv/bin/python scripts/build_phase_h_catalogs.py \
    --min-rows 100 --top-n 2000 --stratify-by-cohort
```

Expected on a 16 GB MIMIC-IV duckdb: ~847 itemids, ~1.4 MB JSON,
~10–15 minutes compute. Adds the unstratified `"all"` bucket per
itemid plus per-cohort buckets where there are at least 30 rows.

You can also run without `--stratify-by-cohort` to get the legacy
flat schema (Tier B) — useful if your duckdb doesn't have a
populated `diagnoses_icd` table.

## Tier 1 — instant wins (no install)

### PubMed: optional API key for higher rate limits

The default backend works. Adding an NCBI API key raises the rate
limit from 3 req/s to 10 req/s — recommended if you'll run the smoke
script repeatedly or expect bursty critic activity.

```bash
# .env
NCBI_API_KEY=your_key
```

Get a key (free, instant) at
https://account.ncbi.nlm.nih.gov/ → Account Settings → API Key
Management → Create.

### PubMed: switch to Anthropic-hosted MCP backend

Better semantic search than NCBI E-utilities for the same query
(returns more relevant PMIDs for clinical phrasing). Uses your
existing `ANTHROPIC_API_KEY` — no additional credentials.

```bash
# .env
PUBMED_BACKEND=mcp_anthropic
```

⚠️ **ZDR caveat.** Per the source comment in
`src/conversational/health_evidence/tools.py`, the Anthropic-hosted
PubMed MCP is **not ZDR-eligible** per Anthropic's Dec 2025 BAA. Don't
use it on conversations that have already touched MIMIC content. If
that's a binding constraint, leave `PUBMED_BACKEND` unset (defaults to
`direct`) or stand up the self-hosted variant below.

### PubMed: self-hosted MCP backend

The audit-grade option for production. Out of scope for getting
started — you'd deploy your own MCP server (e.g., a thin wrapper
around NCBI E-utilities running in your VPC).

```bash
# .env
PUBMED_BACKEND=mcp_self_host
PUBMED_MCP_URL=https://your-host/mcp
```

## Tier 2 — easy (npx auto-installs the server)

These backends auto-pull their MCP server via `npx` on first call.
You need Node.js installed (`which npx` should resolve).

### OpenFDA — drug labels

Auto-pulls `openfda-mcp-server`. Default works without auth (~1k
req/day rate limit).

```bash
# .env (optional but recommended — raises to ~120k req/day)
OPENFDA_API_KEY=your_key
```

Get a key (free, instant): https://open.fda.gov/apis/authentication/

Smoke test:
```bash
.venv/bin/python -c "from src.conversational.health_evidence.tools import openfda_drug_label; print(openfda_drug_label(drug_name='Levophed'))"
```

First call is slow (npx pulls the package, ~30s); subsequent calls
are fast.

### ClinicalTrials.gov — registered studies

Auto-pulls `clinicaltrialsgov-mcp-server` via `bunx` (preferred) or
`npx` (fallback). No auth required.

```bash
# Nothing to add to .env in the default case.
# To pin a specific launcher / version:
# CLINICALTRIALS_MCP_COMMAND="npx -y clinicaltrialsgov-mcp-server@latest"
```

Smoke test:
```bash
.venv/bin/python -c "from src.conversational.health_evidence.tools import trials_search; print(trials_search(query='sepsis lactate', max_results=2))"
```

## Tier 2 — hosted, zero-install (continued)

### OMOPHub — RxNorm + ICD-10 + cross-vocab code mapping

OMOPHub is the OHDSI-ecosystem managed vocabulary service. The
default integration uses their **hosted MCP endpoint** at
`https://mcp.omophub.com` — zero infrastructure, just an API key
(prefix `oh_`) from https://dashboard.omophub.com.

```bash
# .env
OMOPHUB_API_KEY=oh_your_key_here
# OMOPHUB_MCP_URL is optional — defaults to https://mcp.omophub.com.
# Override only for self-hosted Docker (see below).
```

This single integration powers FOUR critic tools:
- `rxnorm_lookup(drug_name)` — RxNorm via `search_concepts(vocabulary_ids="RxNorm")`
- `code_map(source_vocabulary, source_code, target_vocabulary)` — 2-step pivot through `get_concept_by_code` → `explore_concept`
- `icd_lookup(query, version="10")` — ICD-10-CM via `search_concepts(vocabulary_ids="ICD10CM")`
- `icd_autocode(text, version="10")` — ICD-10-CM via `semantic_search(vocabulary_ids="ICD10CM")`

**Auth:** the hosted endpoint authenticates the client request via
`Authorization: Bearer <key>` headers. Our McpClient handles this
automatically — `_omophub_config()` reads `OMOPHUB_API_KEY` and sets
the header. Never hardcoded; never logged.

**ICD-11 limitation:** OMOPHub doesn't carry ICD-11. Calling
`icd_lookup(query=..., version="11")` returns `unavailable` with a
note. For ICD-11 support, set `ICD_MCP_URL` to a self-hosted MCP
that wraps the WHO ICD-API (out of scope; the legacy code path is
preserved for users who do this).

**Self-hosted Docker (optional, advanced):** if you have offline /
privacy / rate-limit requirements, OMOPHub also publishes a Docker
image:

```yaml
# docker-compose.yml (optional — the hosted path is the default)
services:
  omophub-mcp:
    image: omophub/omophub-mcp
    ports: ["3100:3100"]
    env_file: .env
```

Then override the URL: `OMOPHUB_MCP_URL=http://localhost:3100/mcp` in
`.env`. The same `Authorization: Bearer` header is sent (the
container ignores it in favour of its own server-side env auth, but
the header is harmless).

What you lose by skipping this: drug-name disambiguation
(`norepinephrine` vs `Levophed`), ICD-10 lookups (`heart failure` →
`I50.x`), ICD autocoding for clinical narratives, and cross-vocab
code mapping (ICD ↔ SNOMED ↔ RxNorm). The critic falls back to
PubMed for drug-name lookups — usable, just less precise.

## Tier 4 — hard (install + license)

### Hermes — SNOMED CT search and ECL expansion

Hardest tier-1 setup. Hermes is a Java app (you already have OpenJDK
on macOS via Homebrew). The hard part is the SNOMED CT release file —
free for member-state users via national license, otherwise requires
an IHTSDO affiliate license.

```bash
# 1. Install Hermes
brew install --cask wardle/hermes/hermes
# Or download a release: https://github.com/wardle/hermes/releases/latest

# 2. Get SNOMED CT International Edition (free for member states)
#    US:  https://www.nlm.nih.gov/healthit/snomedct/  (NLM UMLS)
#    UK:  https://isd.digital.nhs.uk/trud             (NHS TRUD)
#    Other: https://www.snomed.org/get-snomed

# 3. Build the Hermes index (one-time, ~5 min on the international release)
hermes --db /path/to/snomed.db import /path/to/snomed-release.zip
hermes --db /path/to/snomed.db compact

# 4. .env
HERMES_MCP_COMMAND=hermes
HERMES_MCP_DB=/path/to/snomed.db
```

What you lose by skipping: cohort-canonicalization (e.g., resolving
"septic shock" → SNOMED 76571007 to verify the cohort definition).
The critic falls back to free-text reasoning, which is acceptable for
common cohorts.

## Tier 5 — ICD-11 self-hosted MCP (optional, advanced)

ICD-10 is now covered by OMOPHub (Tier 2 above). For **ICD-11
specifically**, OMOPHub doesn't ship the vocabulary, so users who
need ICD-11 must self-host:

1. Register for ICD-API access at https://icd.who.int/icdapi
2. Run a self-hosted MCP server that wraps the WHO ICD-11 API (no
   public reference implementation; build your own — the
   `mcp_servers/bq_validator/server.py` template in this repo is a
   good FastMCP starting point).

```bash
# .env (only when self-hosting ICD-11)
ICD_MCP_URL=https://your-icd-mcp/mcp
```

When `ICD_MCP_URL` is set, our `icd_lookup` / `icd_autocode` tools
use the legacy dialect (calling tool names `lookup` / `autocode`
with the legacy payload shape). Your MCP needs to expose those two
tools. ICD-11 versions are supported via this path.

When `ICD_MCP_URL` is unset but `OMOPHUB_API_KEY` is set, the tools
route through OMOPHub for ICD-10 only. ICD-11 returns `unavailable`
with a note pointing at this self-host path.

What you lose by skipping ICD entirely: the critic falls back to
free-text reasoning over the diagnosis name, which is usually fine
for plausibility checking but doesn't verify codes.

## Recommended setup order

For a 10-minute "get a real-backend smoke pass" session:

```bash
# 1. NCBI key (60 seconds — improves PubMed)
echo 'NCBI_API_KEY=YOUR_NCBI_KEY' >> .env

# 2. OpenFDA key (60 seconds — required only for high volume)
echo 'OPENFDA_API_KEY=YOUR_OPENFDA_KEY' >> .env

# 3. Re-source .env and run the smoke
set -a && source .env && set +a
.venv/bin/python scripts/critic_evidence_smoke.py
```

Expected outcome: 5 of 8 backends green (PubMed, LOINC, MIMIC,
OpenFDA, ClinicalTrials). The remaining 3 (SNOMED, RxNorm, ICD) stay
`skipped` until you do the harder-tier setup.

## Verifying a single backend in isolation

Each tool can be called directly to isolate a setup problem from a
critic-prompt problem:

```bash
.venv/bin/python <<'PY'
from src.conversational.health_evidence.tools import (
    pubmed_search,
    openfda_drug_label,
    trials_search,
    rxnorm_lookup,
    snomed_search,
    icd_lookup,
)
print("pubmed:", pubmed_search(query="lactate sepsis", max_results=1))
print("openfda:", openfda_drug_label(drug_name="Levophed"))
print("trials:", trials_search(query="sepsis", max_results=1))
print("rxnorm:", rxnorm_lookup(drug_name="norepinephrine"))
print("snomed:", snomed_search(term="septic shock"))
print("icd:", icd_lookup(query="heart failure"))
PY
```

Each line returns either `{"status": "ok", "results": [...]}` or
`{"status": "unavailable", "error": "..."}`. The error string tells
you exactly which env var or binary is missing — match it against
the tier where that tool is documented above.

## When you're done

Drop into the smoke test guide
(`docs/phase-h-smoke-test.md`) — it walks through manual scenarios
that exercise the configured backends through the critic, not just
the bare tool call.
