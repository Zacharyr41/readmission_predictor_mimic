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
  `data/processed/lab_distributions.json`.

If both JSONs are present and you have internet for NCBI, you're at
the floor of what Phase H needs. The smoke test passes its "at least
one backend invoked" gate at this point.

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

## Tier 3 — moderate (paid or licensed service)

### OMOPHub — RxNorm + cross-vocab code mapping

OMOPHub is the OHDSI-ecosystem managed vocabulary service. They sell
hosted access at https://omophub.com. Self-hosting is possible if you
have an Athena vocabulary dump.

```bash
# .env
OMOPHUB_MCP_URL=https://your-omophub-instance/mcp
OMOPHUB_API_KEY=your_key
```

What you lose by skipping this: drug-name disambiguation (`norepinephrine`
vs `Levophed` vs `noradrenaline` all resolving to RXCUI 7980) and
vocab cross-mapping (`code_map`). The critic falls back to PubMed for
drug-name lookups — usable, just less precise.

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

## Tier 5 — needs self-hosting (WHO licensing)

### ICD MCP — ICD-10 / ICD-11 code lookup

WHO's ICD-11 API requires registered access; ICD-10 is public-domain
but there's no first-party MCP server we know of. You'd need to:

1. Register for ICD-API access at https://icd.who.int/icdapi
2. Run a self-hosted MCP server that wraps the WHO API (no public
   reference implementation; build your own).

```bash
# .env (only when you have a server running)
ICD_MCP_URL=https://your-icd-mcp/mcp
```

What you lose by skipping: ICD code verification when an answer cites
a diagnosis code. The critic falls back to free-text reasoning over
the diagnosis name, which is usually fine for plausibility checking.

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
