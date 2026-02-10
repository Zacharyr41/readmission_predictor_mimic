# Architecture

This document describes the architecture of the Temporal Knowledge Graph-Based Hospital Readmission Prediction system.

## Research Foundation

This implementation is based on the conceptual modeling approach for temporal knowledge graphs presented by Vannieuwenhuyze, Mimouni, and Du Mouza in ["A Conceptual Model for Discovering Implicit Temporal Knowledge in Clinical Data"](https://doi.org/10.1007/978-3-032-08620-4_6) (ER 2025). The original implementation is available at [github.com/avannieuwenhuyze/clinical-tkg-cmls2025](https://github.com/avannieuwenhuyze/clinical-tkg-cmls2025).

## System Overview

The system is organized as a 5-layer pipeline that transforms raw MIMIC-IV data into trained machine learning models for predicting 30-day hospital readmission.

```mermaid
flowchart LR
    subgraph Input
        MIMIC[MIMIC-IV CSVs]
    end

    subgraph "Layer 1: Ingestion"
        L1[DuckDB Database]
    end

    subgraph "Layer 2: Graph Construction"
        L2[RDF Knowledge Graph]
    end

    subgraph "Layer 3: Analysis"
        L3[Graph Metrics Report]
    end

    subgraph "Layer 4: Features"
        L4[Feature Matrix]
    end

    subgraph "Layer 5: Prediction"
        L5A[Logistic Regression]
        L5B[XGBoost]
    end

    MIMIC --> L1
    L1 --> L2
    L2 --> L3
    L2 --> L4
    L4 --> L5A
    L4 --> L5B
```

## Layer Details

### Layer 1: Data Ingestion

**Purpose**: Load MIMIC-IV data into DuckDB for efficient SQL querying. Supports two data sources:
- **Local CSVs** (`DATA_SOURCE=local`): Auto-discovers and loads all CSV/CSV.GZ files from `hosp/` and `icu/` directories
- **Google BigQuery** (`DATA_SOURCE=bigquery`): Pulls data directly from `physionet-data.mimiciv_hosp` / `physionet-data.mimiciv_icu` using two-phase loading — small/dimension tables are loaded in full, then large tables (labevents, chartevents, microbiologyevents, prescriptions) are filtered by cohort subject IDs to minimize data transfer

**Modules**:
- `src/ingestion/__init__.py` - Dispatch function `load_mimic_data()` routes to the appropriate loader
- `src/ingestion/mimic_loader.py` - Local CSV → DuckDB loader
- `src/ingestion/bigquery_loader.py` - BigQuery → DuckDB loader (two-phase with cohort filtering)
- `src/ingestion/derived_tables.py` - Creates age, readmission labels, and cohort selection tables

**Input**: MIMIC-IV directory (local) or BigQuery project (cloud)
**Output**: `data/processed/mimiciv.duckdb`

```mermaid
flowchart TB
    subgraph "MIMIC-IV Directory"
        hosp[hosp/]
        icu[icu/]
    end

    subgraph "mimic_loader.py"
        discover[Discover Files]
        load[Load to DuckDB]
    end

    subgraph "derived_tables.py"
        age[create_age_table]
        labels[create_readmission_labels]
        cohort[select_neurology_cohort]
    end

    subgraph Output
        db[(DuckDB)]
    end

    hosp --> discover
    icu --> discover
    discover --> load
    load --> db
    db --> age
    db --> labels
    db --> cohort
    age --> db
    labels --> db
```

**Key Tables Loaded** (31 total):
- `patients`, `admissions`, `icustays` - Core patient data
- `diagnoses_icd`, `d_icd_diagnoses` - Diagnosis codes
- `labevents`, `d_labitems` - Lab results
- `chartevents`, `d_items` - Vital signs and observations
- `prescriptions` - Medications
- `microbiologyevents` - Culture results

**Derived Tables**:
- `age` - Age at admission (computed from anchor values)
- `readmission_labels` - 30/60-day readmission flags

### Layer 2: Graph Construction

**Purpose**: Build an OWL-Time compliant RDF knowledge graph from the relational data.

**Modules**:
- `src/graph_construction/pipeline.py` - Orchestrates the graph build process
- `src/graph_construction/ontology.py` - Initializes namespaces and loads base ontologies
- `src/graph_construction/patient_writer.py` - Creates Patient and HospitalAdmission nodes
- `src/graph_construction/event_writers.py` - Creates clinical event nodes
- `src/graph_construction/temporal/allen_relations.py` - Computes Allen interval relations

**Input**: DuckDB database + ontology files
**Output**: `data/processed/knowledge_graph.rdf`

```mermaid
flowchart TB
    subgraph Input
        db[(DuckDB)]
        onto[Ontology Files]
    end

    subgraph "pipeline.py"
        init[Initialize Graph]
        process[Process Patients]
    end

    subgraph Writers
        pw[patient_writer]
        ew[event_writers]
        ar[allen_relations]
    end

    subgraph Output
        rdf[RDF Graph]
    end

    db --> init
    onto --> init
    init --> process
    process --> pw
    process --> ew
    process --> ar
    pw --> rdf
    ew --> rdf
    ar --> rdf
```

### Layer 3: Graph Analysis

**Purpose**: Generate statistics and structure analysis of the knowledge graph.

**Modules**:
- `src/graph_analysis/analysis.py` - SPARQL queries for graph metrics
- `src/graph_analysis/rdf_to_networkx.py` - Converts RDF to NetworkX for algorithms

**Input**: RDF knowledge graph
**Output**: `outputs/reports/graph_analysis.md`

### Layer 4: Feature Extraction

**Purpose**: Extract ML-ready features from the knowledge graph using SPARQL queries.

**Modules**:
- `src/feature_extraction/tabular_features.py` - Demographics, labs, vitals, medications, diagnoses
- `src/feature_extraction/graph_features.py` - Temporal relations, graph structure metrics
- `src/feature_extraction/feature_builder.py` - Combines all features into single matrix

**Input**: RDF knowledge graph
**Output**: `data/features/feature_matrix.parquet`

```mermaid
flowchart LR
    subgraph Input
        rdf[RDF Graph]
    end

    subgraph "tabular_features.py"
        demo[Demographics]
        stay[Stay Features]
        lab[Lab Summary]
        vital[Vital Summary]
        med[Medications]
        dx[Diagnoses]
    end

    subgraph "graph_features.py"
        temp[Temporal Features]
        struct[Structure Features]
    end

    subgraph "feature_builder.py"
        merge[Merge Features]
        fill[Fill Missing]
    end

    subgraph Output
        matrix[Feature Matrix]
    end

    rdf --> demo
    rdf --> stay
    rdf --> lab
    rdf --> vital
    rdf --> med
    rdf --> dx
    rdf --> temp
    rdf --> struct

    demo --> merge
    stay --> merge
    lab --> merge
    vital --> merge
    med --> merge
    dx --> merge
    temp --> merge
    struct --> merge
    merge --> fill
    fill --> matrix
```

### Layer 5: Prediction

**Purpose**: Train and evaluate classification models.

**Modules**:
- `src/prediction/split.py` - Patient-level train/val/test splitting
- `src/prediction/model.py` - Model training (LR, XGBoost)
- `src/prediction/evaluate.py` - Metrics computation and report generation

**Input**: Feature matrix
**Output**: Models and evaluation reports

```mermaid
flowchart TB
    subgraph Input
        matrix[Feature Matrix]
    end

    subgraph "split.py"
        split[Patient-Level Split]
    end

    subgraph "model.py"
        lr[Logistic Regression]
        xgb[XGBoost]
    end

    subgraph "evaluate.py"
        eval[Compute Metrics]
        report[Generate Report]
    end

    subgraph Output
        models[Trained Models]
        reports[Evaluation Reports]
    end

    matrix --> split
    split --> lr
    split --> xgb
    lr --> eval
    xgb --> eval
    eval --> report
    lr --> models
    xgb --> models
    report --> reports
```

## Data Flow

```mermaid
sequenceDiagram
    participant CSV as MIMIC-IV CSVs
    participant Duck as DuckDB
    participant RDF as RDF Graph
    participant Feat as Feature Matrix
    participant Model as ML Models

    CSV->>Duck: load_mimic_to_duckdb()
    Duck->>Duck: create_age_table()
    Duck->>Duck: create_readmission_labels()
    Duck->>Duck: select_neurology_cohort()

    Duck->>RDF: build_graph()
    Note over RDF: Write patients, admissions,<br/>ICU stays, events
    RDF->>RDF: compute_allen_relations()

    RDF->>Feat: build_feature_matrix()
    Note over Feat: SPARQL queries extract<br/>tabular and graph features

    Feat->>Model: patient_level_split()
    Feat->>Model: train_model()
    Model->>Model: evaluate_model()
    Model->>Model: generate_evaluation_report()
```

## Entity-Relationship Diagram

The RDF graph models clinical data using OWL-Time temporal ontology:

```mermaid
erDiagram
    Patient ||--o{ HospitalAdmission : hasAdmission
    Patient {
        int subject_id
        string gender
        int age
    }

    HospitalAdmission ||--o{ ICUStay : containsICUStay
    HospitalAdmission ||--o{ DiagnosisEvent : hasDiagnosis
    HospitalAdmission {
        int hadm_id
        datetime admittime
        datetime dischtime
        string admission_type
        boolean readmitted_30d
        boolean readmitted_60d
    }

    ICUStay ||--o{ ICUDay : hasICUDay
    ICUStay ||--o{ BioMarkerEvent : hasEvent
    ICUStay ||--o{ ClinicalSignEvent : hasEvent
    ICUStay ||--o{ MicrobiologyEvent : hasEvent
    ICUStay ||--o{ AntibioticEvent : hasEvent
    ICUStay {
        int stay_id
        datetime intime
        datetime outtime
        float los_days
    }

    ICUDay {
        int day_number
        datetime start
        datetime end
    }

    BioMarkerEvent {
        string biomarker_type
        float value
        string unit
        datetime charttime
    }

    ClinicalSignEvent {
        string vital_name
        float value
        datetime charttime
    }

    DiagnosisEvent {
        string icd_code
        int icd_version
        int seq_num
    }
```

## Temporal Modeling

Events are modeled as OWL-Time intervals or instants with Allen temporal relations:

```mermaid
flowchart LR
    subgraph "Allen Relations"
        before["A before B<br/>A---| |---B"]
        meets["A meets B<br/>A---|B---"]
        overlaps["A overlaps B<br/>A---|--B---"]
        starts["A starts B<br/>|A---|<br/>|--B---|"]
        during["A during B<br/>|--A--|<br/>|---B---|"]
        finishes["A finishes B<br/>|---A|<br/>|---B---|"]
    end
```

Each relation is represented using OWL-Time predicates:
- `time:before` - A ends before B starts
- `time:intervalMeets` - A ends exactly when B starts
- `time:intervalOverlaps` - A starts before B, A ends during B
- `time:intervalStarts` - Same start, A ends first
- `time:inside` (during) - A entirely within B
- `time:intervalFinishes` - Same end, A starts later

## Module Dependencies

```mermaid
flowchart TD
    subgraph Config
        settings[config/settings.py]
    end

    subgraph Layer1
        loader[mimic_loader]
        derived[derived_tables]
    end

    subgraph Layer2
        pipeline[pipeline]
        ontology[ontology]
        patient[patient_writer]
        events[event_writers]
        allen[allen_relations]
    end

    subgraph Layer3
        analysis[analysis]
        nx[rdf_to_networkx]
    end

    subgraph Layer4
        builder[feature_builder]
        tabular[tabular_features]
        graph_feat[graph_features]
    end

    subgraph Layer5
        split[split]
        model[model]
        evaluate[evaluate]
    end

    subgraph Main
        main[main.py]
    end

    settings --> main
    main --> loader
    main --> pipeline
    main --> analysis
    main --> builder
    main --> split
    main --> model
    main --> evaluate

    loader --> derived
    pipeline --> ontology
    pipeline --> patient
    pipeline --> events
    pipeline --> allen
    pipeline --> derived

    analysis --> nx

    builder --> tabular
    builder --> graph_feat
    graph_feat --> nx

    tabular --> ontology
    graph_feat --> ontology
```

## Performance Considerations

### Memory Usage

| Stage | Memory Pattern | Optimization |
|-------|---------------|--------------|
| Ingestion | Streaming | DuckDB handles large files |
| Graph Build | Linear in patients | Process one patient at a time |
| Allen Relations | O(n²) per ICU stay | Use `--skip-allen` flag |
| Feature Extraction | SPARQL queries | Batched by admission |
| Model Training | Full matrix in memory | Sparse representation |

### Processing Time

Approximate times for 1000 patients (16GB RAM machine):

| Stage | Time |
|-------|------|
| Ingestion (first run) | 5-10 min |
| Graph Construction | 2-5 min |
| Allen Relations | 10-30 min |
| Feature Extraction | 1-2 min |
| Model Training | < 1 min |

Use `--skip-allen` to reduce graph construction time by 80%+.

### Parallelization

Current implementation is single-threaded. Potential parallelization points:
- Patient processing in graph construction
- SPARQL queries in feature extraction
- Cross-validation folds in model training
