# Readmission Predictor MIMIC

Temporal knowledge graph for hospital readmission prediction using MIMIC-IV data.

## Setup

```bash
make setup
cp .env.example .env
# Edit .env with your paths
```

## Testing

```bash
make test
```

## Pipeline

```bash
make all  # Run full pipeline: ingest -> graph -> features -> train
```
