#!/usr/bin/env bash
#
# Setup and verify BigQuery access for MIMIC-IV data.
#
# Prerequisites:
#   1. Google Cloud SDK installed (https://cloud.google.com/sdk/docs/install)
#   2. PhysioNet credentialed access (https://physionet.org/settings/credentialing/)
#   3. PhysioNet account linked to your Google account
#      (https://physionet.org/settings/cloud/)
#   4. A GCP project with the BigQuery API enabled
#
# Usage:
#   ./scripts/setup_bigquery.sh [your-gcp-project-id]
#
# The project ID is resolved in this order:
#   1. CLI argument
#   2. BIGQUERY_PROJECT in .env
#   3. gcloud config default project
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!!]${NC} $1"; }
fail()  { echo -e "${RED}[FAIL]${NC} $1"; }

# Read BIGQUERY_PROJECT from .env if it exists
_read_env_project() {
    local env_file="$PROJECT_ROOT/.env"
    if [ -f "$env_file" ]; then
        grep -E "^BIGQUERY_PROJECT=" "$env_file" 2>/dev/null | head -1 | cut -d= -f2 | tr -d '[:space:]' || true
    fi
}

PROJECT_ID="${1:-}"

echo "=========================================="
echo " BigQuery Setup for MIMIC-IV"
echo "=========================================="
echo ""

# ── Step 1: Check gcloud CLI ──────────────────────────────────────────────

echo "Step 1: Checking Google Cloud SDK..."
if ! command -v gcloud &> /dev/null; then
    fail "gcloud CLI not found."
    echo "  Install it from: https://cloud.google.com/sdk/docs/install"
    echo "  On macOS: brew install --cask google-cloud-sdk"
    exit 1
fi
info "gcloud CLI found: $(gcloud version 2>/dev/null | head -1)"

# ── Step 2: Check authentication ─────────────────────────────────────────

echo ""
echo "Step 2: Checking authentication..."
if ! gcloud auth application-default print-access-token &> /dev/null; then
    warn "No Application Default Credentials found."
    echo "  Running: gcloud auth application-default login"
    echo ""
    gcloud auth application-default login
    echo ""
    if ! gcloud auth application-default print-access-token &> /dev/null; then
        fail "Authentication failed. Please try again."
        exit 1
    fi
fi
info "Application Default Credentials are set."

# ── Step 3: Determine project ID ─────────────────────────────────────────

echo ""
echo "Step 3: Checking GCP project..."
if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID=$(_read_env_project)
    if [ -n "$PROJECT_ID" ]; then
        info "Read BIGQUERY_PROJECT=$PROJECT_ID from .env"
    fi
fi
if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null || true)
fi

if [ -z "$PROJECT_ID" ]; then
    fail "No GCP project ID specified."
    echo "  Either pass it as an argument:"
    echo "    ./scripts/setup_bigquery.sh your-project-id"
    echo "  Or set BIGQUERY_PROJECT in .env"
    echo "  Or set a gcloud default:"
    echo "    gcloud config set project your-project-id"
    exit 1
fi
info "Using GCP project: $PROJECT_ID"

# ── Step 4: Verify BigQuery API is enabled ────────────────────────────────

echo ""
echo "Step 4: Checking BigQuery API..."
if gcloud services list --enabled --project="$PROJECT_ID" 2>/dev/null | grep -q "bigquery.googleapis.com"; then
    info "BigQuery API is enabled."
else
    warn "BigQuery API may not be enabled. Attempting to enable..."
    gcloud services enable bigquery.googleapis.com --project="$PROJECT_ID" 2>/dev/null || {
        fail "Could not enable BigQuery API. Enable it manually at:"
        echo "  https://console.cloud.google.com/apis/library/bigquery.googleapis.com?project=$PROJECT_ID"
        exit 1
    }
    info "BigQuery API enabled."
fi

# ── Step 5: Verify PhysioNet MIMIC-IV access ─────────────────────────────

echo ""
echo "Step 5: Verifying access to MIMIC-IV on BigQuery..."
echo "  Running test query: SELECT COUNT(*) FROM physionet-data.mimiciv_3_1_hosp.patients"

RESULT=$(bq --project_id="$PROJECT_ID" query --use_legacy_sql=false --format=csv \
    "SELECT COUNT(*) AS n FROM \`physionet-data.mimiciv_3_1_hosp.patients\` LIMIT 1" 2>&1) || {
    fail "Cannot query MIMIC-IV tables."
    echo ""
    echo "  This usually means one of:"
    echo "    1. Your PhysioNet account is not linked to your Google account"
    echo "       -> Go to https://physionet.org/settings/cloud/"
    echo "       -> Link your Google account under 'Google BigQuery'"
    echo ""
    echo "    2. You haven't signed the MIMIC-IV data use agreement"
    echo "       -> Go to https://physionet.org/content/mimiciv/"
    echo "       -> Click 'Request access' and complete credentialing"
    echo ""
    echo "    3. Your GCP project doesn't have billing enabled"
    echo "       -> BigQuery queries require a billing-enabled project"
    echo ""
    echo "  Error details:"
    echo "  $RESULT"
    exit 1
}

PATIENT_COUNT=$(echo "$RESULT" | tail -1 | tr -d '[:space:]')
info "MIMIC-IV access verified! patients table has $PATIENT_COUNT rows."

# ── Step 6: Update .env file ─────────────────────────────────────────────

echo ""
echo "Step 6: Updating .env configuration..."

ENV_FILE="$PROJECT_ROOT/.env"

if [ ! -f "$ENV_FILE" ]; then
    if [ -f "$PROJECT_ROOT/.env.example" ]; then
        cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
        info "Created .env from .env.example"
    else
        touch "$ENV_FILE"
        info "Created empty .env"
    fi
fi

# Update or add DATA_SOURCE
if grep -q "^DATA_SOURCE=" "$ENV_FILE" 2>/dev/null; then
    sed -i.bak "s/^DATA_SOURCE=.*/DATA_SOURCE=bigquery/" "$ENV_FILE"
else
    echo "DATA_SOURCE=bigquery" >> "$ENV_FILE"
fi

# Update or add BIGQUERY_PROJECT
if grep -q "^BIGQUERY_PROJECT=" "$ENV_FILE" 2>/dev/null; then
    sed -i.bak "s/^BIGQUERY_PROJECT=.*/BIGQUERY_PROJECT=$PROJECT_ID/" "$ENV_FILE"
else
    echo "BIGQUERY_PROJECT=$PROJECT_ID" >> "$ENV_FILE"
fi

rm -f "$ENV_FILE.bak"
info "Set DATA_SOURCE=bigquery and BIGQUERY_PROJECT=$PROJECT_ID in .env"

# ── Done ──────────────────────────────────────────────────────────────────

echo ""
echo "=========================================="
echo -e " ${GREEN}Setup complete!${NC}"
echo "=========================================="
echo ""
echo "You can now run the pipeline with BigQuery:"
echo "  .venv/bin/python -m src.main --run-ingestion --patients-limit 50 --skip-allen -v"
echo ""
echo "Or run ingestion only:"
echo "  .venv/bin/python -m src.ingestion --data-source bigquery"
echo ""
