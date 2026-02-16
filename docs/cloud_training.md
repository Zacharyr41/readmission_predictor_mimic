# Cloud Training on Vertex AI

Run the full readmission-prediction pipeline (BigQuery ingestion through GNN training) on a GPU-equipped Vertex AI instance. Submit the job and close your laptop -- results upload to GCS automatically.

## Prerequisites

- **gcloud CLI** installed and authenticated (`gcloud auth login`)
- **Docker** running locally (used to build and push the training image)
- **GCP project** with billing enabled
- **PhysioNet BigQuery access**: your Google account must be credentialed for `physionet-data` BigQuery tables (see [PhysioNet instructions](https://physionet.org/about/database/#access))
- **Application Default Credentials** for BigQuery access:

```bash
gcloud auth application-default login
```

This writes credentials to `~/.config/gcloud/application_default_credentials.json`, which the setup script stores in Secret Manager so the Vertex AI job can reach BigQuery.

## One-time setup

```bash
./scripts/setup_cloud_training.sh
```

Run this once per project. It is idempotent -- safe to re-run.

### What it does

1. **Enables GCP APIs**: Compute Engine, Vertex AI, Artifact Registry, Cloud Storage, Secret Manager, BigQuery.
2. **Creates an Artifact Registry repository** (`readmission-training`) for Docker images.
3. **Creates a GCS bucket** (`<PROJECT_ID>-readmission-training`) for storing training outputs.
4. **Stores your ADC credentials** in Secret Manager (secret name: `physionet-adc`) so the Vertex AI container can authenticate to BigQuery at runtime.
5. **Grants IAM roles** to the Compute Engine default service account:
   - `roles/bigquery.jobUser` -- run BigQuery queries
   - `roles/storage.objectAdmin` -- write results to GCS
   - `roles/secretmanager.secretAccessor` -- read the ADC secret

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `GCP_PROJECT` | `gcloud config get-value project` | GCP project ID |
| `GCP_REGION` | `us-central1` | Region for all resources |

Example with explicit project:

```bash
GCP_PROJECT=my-project-id ./scripts/setup_cloud_training.sh
```

## Submitting a training job

```bash
./scripts/cloud_train.sh [OPTIONS]
```

This builds a Docker image from the repo root, pushes it to Artifact Registry, and submits a Vertex AI custom training job with an NVIDIA T4 GPU.

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--experiment NAME` | `E6_full_model` | Experiment to run (see list below) |
| `--patients-limit N` | `0` (no limit) | Limit cohort size for faster iteration |
| `--machine-type TYPE` | `n1-highmem-4` | Vertex AI machine type |
| `--seed N` | `42` | Random seed |
| `--run-all` | off | Run all 6 ablation experiments sequentially |
| `--no-skip-allen` | off | Compute Allen temporal relations (significantly slower) |

### Available experiments

| Name | Description |
|---|---|
| `E1_mlp_baseline` | Floor performance baseline: projection + classifier only |
| `E2_mlp_sapbert` | Value of SapBERT priors: MLP with enriched features |
| `E3_transformer_only` | Does graph structure help over MLP baseline? |
| `E4_transformer_temporal` | Temporal encoding on top of transformer |
| `E5_full_no_temporal` | Cross-view denoising (diffusion without temporal) |
| `E6_full_model` | Complete TD4DD model with all components |

### Examples

Quick smoke test with 50 patients:

```bash
./scripts/cloud_train.sh --patients-limit 50
```

Full training run (default experiment, all patients):

```bash
./scripts/cloud_train.sh
```

Run a specific experiment with a different seed:

```bash
./scripts/cloud_train.sh --experiment E3_transformer_only --seed 123
```

Run all 6 ablation experiments and produce a comparison table:

```bash
./scripts/cloud_train.sh --run-all
```

Use a larger machine:

```bash
./scripts/cloud_train.sh --machine-type n1-highmem-8
```

Explicit project and region:

```bash
GCP_PROJECT=my-project GCP_REGION=us-east1 ./scripts/cloud_train.sh --patients-limit 100
```

## Monitoring jobs

List recent jobs:

```bash
gcloud ai custom-jobs list --project=PROJECT_ID --region=us-central1
```

Stream logs from a running job:

```bash
gcloud ai custom-jobs stream-logs JOB_ID --project=PROJECT_ID --region=us-central1
```

Get the job ID from the `list` output or from the submission output printed by `cloud_train.sh`.

Describe a specific job (status, start/end time, errors):

```bash
gcloud ai custom-jobs describe JOB_ID --project=PROJECT_ID --region=us-central1
```

Cancel a running job:

```bash
gcloud ai custom-jobs cancel JOB_ID --project=PROJECT_ID --region=us-central1
```

## Downloading results

The training container automatically uploads its `outputs/` directory to GCS at `gs://<PROJECT_ID>-readmission-training/runs/<RUN_ID>/`.

Download everything:

```bash
gsutil -m cp -r gs://PROJECT_ID-readmission-training/runs/ ./cloud_outputs/
```

Download a specific run:

```bash
gsutil -m cp -r gs://PROJECT_ID-readmission-training/runs/E6_full_model_1700000000/ ./cloud_outputs/
```

List available runs:

```bash
gsutil ls gs://PROJECT_ID-readmission-training/runs/
```

### Output contents

- `outputs/run_info.json` -- run metadata (experiment, seed, patients_limit, elapsed time)
- `outputs/gnn_experiments/<experiment>/` -- model checkpoints, metrics, training logs
- `outputs/comparison.md` -- markdown comparison table (only when `--run-all` is used)

## Cost estimate

All prices are approximate (us-central1, on-demand, as of 2025).

| Component | Spec | Approx. cost/hour |
|---|---|---|
| Machine | `n1-highmem-4` (4 vCPU, 26 GB RAM) | ~$0.24 |
| GPU | NVIDIA T4 (16 GB VRAM) | ~$0.35 |
| **Total compute** | | **~$0.59/hour** |

Additional costs:

- **BigQuery**: the pipeline scans PhysioNet MIMIC-IV tables. A full cohort run typically scans a few GB (~$0.03 at $6.25/TB).
- **Artifact Registry**: image storage is negligible (a few GB at ~$0.10/GB/month).
- **GCS**: output storage is negligible (a few MB per run).

A full single-experiment run typically completes in 30-90 minutes depending on cohort size, costing roughly $0.30-$0.90 in compute. Running all 6 experiments (`--run-all`) takes 3-8 hours.

Use `--patients-limit 50` for development iteration -- these runs complete in a few minutes and cost under $0.10.

## Troubleshooting

### GPU quota not available

**Error**: `RESOURCE_EXHAUSTED` or quota error when submitting the job.

The T4 GPU requires quota in your project. Request it at:
**IAM & Admin > Quotas** -- filter for `NVIDIA_T4_GPUS` in your region.

Alternatively, try a different region:

```bash
GCP_REGION=us-east1 ./scripts/cloud_train.sh
```

### Docker not running

**Error**: `Cannot connect to the Docker daemon` during `cloud_train.sh`.

Start Docker Desktop (or the Docker daemon) before running the script. The script builds the image locally before pushing.

### ADC credentials not found

**Error**: `ADC credentials not found at ~/.config/gcloud/application_default_credentials.json` during setup.

Run the following and re-run setup:

```bash
gcloud auth application-default login
```

### Secret Manager access denied at runtime

**Error**: `PermissionDenied` when the container tries to fetch the `physionet-adc` secret.

Re-run the setup script to ensure IAM roles are granted:

```bash
./scripts/setup_cloud_training.sh
```

If the issue persists, verify the Compute Engine default service account has `roles/secretmanager.secretAccessor`:

```bash
PROJECT_NUMBER=$(gcloud projects describe PROJECT_ID --format='value(projectNumber)')
gcloud projects get-iam-policy PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --format="table(bindings.role)"
```

### BigQuery permission denied

**Error**: `Access Denied` on `physionet-data` tables.

Your ADC credentials must belong to a Google account that has accepted the PhysioNet data use agreement for MIMIC-IV on BigQuery. Verify at [PhysioNet](https://physionet.org/settings/credentialing/).

After fixing credentials, re-run setup to update the secret:

```bash
gcloud auth application-default login
./scripts/setup_cloud_training.sh
```

### Job fails immediately with OOM

The default `n1-highmem-4` provides 26 GB of RAM. For full cohort runs, this is usually sufficient. If you hit memory limits, scale up:

```bash
./scripts/cloud_train.sh --machine-type n1-highmem-8
```

### Docker build fails on torch-sparse / torch-scatter

These packages require matching CUDA and PyTorch versions. The Dockerfile pins `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` and fetches wheels from the corresponding PyG index. If you change the base image, update the `--find-links` URL in the Dockerfile to match.
