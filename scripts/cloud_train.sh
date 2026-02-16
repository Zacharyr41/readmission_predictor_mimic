#!/usr/bin/env bash
# Build Docker image, push to Artifact Registry, and submit a Vertex AI
# custom training job. Fire-and-forget: close your laptop after submitting.
#
# Usage:
#   ./scripts/cloud_train.sh                          # Full run (E6_full_model)
#   ./scripts/cloud_train.sh --patients-limit 50      # Quick test
#   ./scripts/cloud_train.sh --experiment E3_transformer_only
#   ./scripts/cloud_train.sh --run-all                # All 6 ablation experiments
#   ./scripts/cloud_train.sh --seed 123
#   ./scripts/cloud_train.sh --machine-type n1-highmem-8

set -euo pipefail

# ── Defaults ──
EXPERIMENT="E6_full_model"
SEED="42"
PATIENTS_LIMIT="0"
SKIP_ALLEN="1"
RUN_ALL="0"
MACHINE_TYPE="n1-highmem-4"

# ── Parse CLI flags ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --patients-limit)
            PATIENTS_LIMIT="$2"; shift 2 ;;
        --experiment)
            EXPERIMENT="$2"; shift 2 ;;
        --machine-type)
            MACHINE_TYPE="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        --run-all)
            RUN_ALL="1"; shift ;;
        --no-skip-allen)
            SKIP_ALLEN="0"; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --patients-limit N    Limit cohort size (default: 0 = no limit)"
            echo "  --experiment NAME     Experiment to run (default: E6_full_model)"
            echo "  --machine-type TYPE   Vertex AI machine type (default: n1-highmem-4)"
            echo "  --seed N              Random seed (default: 42)"
            echo "  --run-all             Run all 6 ablation experiments"
            echo "  --no-skip-allen       Compute Allen temporal relations (slower)"
            echo "  -h, --help            Show this help"
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Resolve GCP config ──
PROJECT_ID="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
if [[ -z "$PROJECT_ID" ]]; then
    echo "ERROR: Set GCP_PROJECT or run 'gcloud config set project <id>'" >&2
    exit 1
fi
REGION="${GCP_REGION:-us-central1}"
BUCKET="${PROJECT_ID}-readmission-training"
REPO="readmission-training"
IMAGE_TAG="$(date +%Y%m%d-%H%M%S)"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/readmission-gnn:${IMAGE_TAG}"

echo "=== Cloud Training Submission ==="
echo "Project:     $PROJECT_ID"
echo "Region:      $REGION"
echo "Image:       $IMAGE_URI"
echo "Machine:     $MACHINE_TYPE + T4 GPU"
echo "Experiment:  $EXPERIMENT"
echo "Patients:    $PATIENTS_LIMIT (0 = all)"
echo "Seed:        $SEED"
echo "Run all:     $RUN_ALL"
echo ""

# ── Build Docker image ──
echo "Building Docker image..."
docker build -t "$IMAGE_URI" .

# ── Push to Artifact Registry ──
echo "Pushing to Artifact Registry..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
docker push "$IMAGE_URI"

# ── Fetch ADC credentials from Secret Manager ──
echo "Fetching ADC credentials from Secret Manager..."
ADC_B64=$(gcloud secrets versions access latest --secret=physionet-adc --project="$PROJECT_ID" | base64)

# ── Prepare job spec ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
JOB_SPEC=$(mktemp /tmp/vertex_job_spec_XXXXXX)
mv "$JOB_SPEC" "${JOB_SPEC}.yaml"
JOB_SPEC="${JOB_SPEC}.yaml"

sed \
    -e "s|MACHINE_TYPE_PLACEHOLDER|$MACHINE_TYPE|g" \
    -e "s|IMAGE_URI_PLACEHOLDER|$IMAGE_URI|g" \
    -e "s|GCP_PROJECT_PLACEHOLDER|$PROJECT_ID|g" \
    -e "s|GCS_BUCKET_PLACEHOLDER|$BUCKET|g" \
    -e "s|EXPERIMENT_PLACEHOLDER|$EXPERIMENT|g" \
    -e "s|SEED_PLACEHOLDER|$SEED|g" \
    -e "s|PATIENTS_LIMIT_PLACEHOLDER|$PATIENTS_LIMIT|g" \
    -e "s|SKIP_ALLEN_PLACEHOLDER|$SKIP_ALLEN|g" \
    -e "s|RUN_ALL_PLACEHOLDER|$RUN_ALL|g" \
    -e "s|ADC_B64_PLACEHOLDER|$ADC_B64|g" \
    "${REPO_DIR}/cloud/vertex_job_spec.yaml" > "$JOB_SPEC"

JOB_NAME="readmission-$(echo "$EXPERIMENT" | tr '[:upper:]' '[:lower:]')-${IMAGE_TAG}"

# ── Submit Vertex AI custom job ──
echo "Submitting Vertex AI custom job: $JOB_NAME"
gcloud ai custom-jobs create \
    --region="$REGION" \
    --project="$PROJECT_ID" \
    --display-name="$JOB_NAME" \
    --config="$JOB_SPEC"

rm -f "$JOB_SPEC"

echo ""
echo "=== Job submitted ==="
echo ""
echo "Monitor:"
echo "  gcloud ai custom-jobs list --project=$PROJECT_ID --region=$REGION"
echo "  gcloud ai custom-jobs stream-logs <JOB_ID> --project=$PROJECT_ID --region=$REGION"
echo ""
echo "Download results:"
echo "  gsutil -m cp -r gs://$BUCKET/runs/ ./cloud_outputs/"
