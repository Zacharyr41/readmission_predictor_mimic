#!/usr/bin/env bash
# One-time setup for GCP cloud training (idempotent).
#
# Enables APIs, creates resources, and stores ADC credentials in Secret Manager
# so the Vertex AI job can access PhysioNet BigQuery tables.
#
# Usage: ./scripts/setup_cloud_training.sh

set -euo pipefail

# ── Resolve GCP project ──
PROJECT_ID="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
if [[ -z "$PROJECT_ID" ]]; then
    echo "ERROR: Set GCP_PROJECT or run 'gcloud config set project <id>'" >&2
    exit 1
fi
REGION="${GCP_REGION:-us-central1}"
BUCKET="${PROJECT_ID}-readmission-training"
REPO="readmission-training"
SECRET_NAME="physionet-adc"
ADC_PATH="${HOME}/.config/gcloud/application_default_credentials.json"

echo "=== Cloud Training Setup ==="
echo "Project:  $PROJECT_ID"
echo "Region:   $REGION"
echo "Bucket:   gs://$BUCKET"
echo "Registry: $REGION-docker.pkg.dev/$PROJECT_ID/$REPO"
echo ""

# ── Enable APIs ──
echo "Enabling APIs..."
gcloud services enable \
    aiplatform.googleapis.com \
    artifactregistry.googleapis.com \
    storage.googleapis.com \
    secretmanager.googleapis.com \
    bigquery.googleapis.com \
    --project="$PROJECT_ID" --quiet

# ── Create Artifact Registry repository ──
echo "Creating Artifact Registry repo '$REPO'..."
if gcloud artifacts repositories describe "$REPO" \
    --location="$REGION" --project="$PROJECT_ID" &>/dev/null; then
    echo "  Already exists."
else
    gcloud artifacts repositories create "$REPO" \
        --repository-format=docker \
        --location="$REGION" \
        --project="$PROJECT_ID" \
        --description="Readmission predictor training images"
fi

# ── Create GCS bucket ──
echo "Creating GCS bucket 'gs://$BUCKET'..."
if gsutil ls -b "gs://$BUCKET" &>/dev/null; then
    echo "  Already exists."
else
    gsutil mb -l "$REGION" -p "$PROJECT_ID" "gs://$BUCKET"
fi

# ── Store ADC credentials in Secret Manager ──
echo "Storing ADC credentials in Secret Manager as '$SECRET_NAME'..."
if [[ ! -f "$ADC_PATH" ]]; then
    echo "ERROR: ADC credentials not found at $ADC_PATH" >&2
    echo "Run 'gcloud auth application-default login' first." >&2
    exit 1
fi

if gcloud secrets describe "$SECRET_NAME" --project="$PROJECT_ID" &>/dev/null; then
    echo "  Secret exists, adding new version..."
    gcloud secrets versions add "$SECRET_NAME" \
        --data-file="$ADC_PATH" \
        --project="$PROJECT_ID"
else
    gcloud secrets create "$SECRET_NAME" \
        --replication-policy="automatic" \
        --project="$PROJECT_ID"
    gcloud secrets versions add "$SECRET_NAME" \
        --data-file="$ADC_PATH" \
        --project="$PROJECT_ID"
fi

# ── Grant IAM roles to Compute Engine default service account ──
echo "Granting IAM roles to Compute Engine default SA..."
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
CE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

for ROLE in \
    roles/bigquery.jobUser \
    roles/storage.objectAdmin \
    roles/secretmanager.secretAccessor; do
    echo "  $ROLE -> $CE_SA"
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$CE_SA" \
        --role="$ROLE" \
        --condition=None \
        --quiet &>/dev/null
done

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  # Quick test (50 patients):"
echo "  ./scripts/cloud_train.sh --patients-limit 50"
echo ""
echo "  # Full training run:"
echo "  ./scripts/cloud_train.sh"
