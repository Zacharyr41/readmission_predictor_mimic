# API Reference

This document provides a module-by-module API reference for the readmission prediction pipeline.

## Table of Contents

- [Main Pipeline](#main-pipeline)
- [Layer 1: Ingestion](#layer-1-ingestion)
- [Layer 2: Graph Construction](#layer-2-graph-construction)
- [Layer 3: Graph Analysis](#layer-3-graph-analysis)
- [Layer 4: Feature Extraction](#layer-4-feature-extraction)
- [Layer 5: Prediction](#layer-5-prediction)

---

## Main Pipeline

### `src.main`

Main pipeline orchestrator module.

#### `run_pipeline()`

```python
def run_pipeline(
    settings: Settings,
    paths: dict[str, Path] | None = None,
    ontology_dir: Path | None = None,
    skip_ingestion: bool = True,
) -> dict[str, Any]
```

Orchestrate the complete readmission prediction pipeline.

**Args:**
- `settings`: Pipeline configuration settings (from `config.settings.Settings`)
- `paths`: Override default artifact paths (dict mapping name to Path)
- `ontology_dir`: Path to ontology files directory
- `skip_ingestion`: If True, skip CSV loading and use existing DuckDB

**Returns:**
Dictionary containing:
- `cohort_size`: Number of patients in cohort
- `graph_triples`: Number of RDF triples
- `feature_shape`: Tuple of (n_admissions, n_features)
- `metrics`: Dict with model evaluation metrics
- `artifact_paths`: Dict of paths to generated artifacts

**Example:**
```python
from config.settings import Settings
from src.main import run_pipeline

settings = Settings()
result = run_pipeline(settings, skip_ingestion=True)
print(f"AUROC: {result['metrics']['xgb']['auroc']:.4f}")
```

---

## Layer 1: Ingestion

### `src.ingestion.mimic_loader`

MIMIC-IV DuckDB loader module.

#### `load_mimic_to_duckdb()`

```python
def load_mimic_to_duckdb(
    source_dir: Path,
    db_path: Path,
) -> duckdb.DuckDBPyConnection
```

Load all MIMIC-IV CSV/CSV.GZ files into DuckDB.

**Args:**
- `source_dir`: Path to MIMIC-IV directory containing `hosp/` and `icu/` subdirs
- `db_path`: Path where DuckDB database file will be created

**Returns:**
DuckDB connection to the loaded database.

**Example:**
```python
from pathlib import Path
from src.ingestion.mimic_loader import load_mimic_to_duckdb

conn = load_mimic_to_duckdb(
    source_dir=Path("/path/to/mimiciv/3.1"),
    db_path=Path("data/processed/mimiciv.duckdb"),
)
print(conn.execute("SELECT COUNT(*) FROM patients").fetchone())
```

#### `get_loaded_tables()`

```python
def get_loaded_tables(conn: duckdb.DuckDBPyConnection) -> list[str]
```

Get list of all loaded table names.

**Args:**
- `conn`: DuckDB connection

**Returns:**
Sorted list of table names.

---

### `src.ingestion.derived_tables`

Derived tables for MIMIC-IV analysis.

#### `create_age_table()`

```python
def create_age_table(conn: duckdb.DuckDBPyConnection) -> None
```

Create derived age table with age at admission.

Age is computed as: `anchor_age + (YEAR(admittime) - anchor_year)`

**Args:**
- `conn`: DuckDB connection with `patients` and `admissions` tables loaded

#### `create_readmission_labels()`

```python
def create_readmission_labels(
    conn: duckdb.DuckDBPyConnection,
    windows: list[int] | None = None,
) -> None
```

Create readmission labels table.

Uses `LEAD()` window function to find next admission for each patient.
Excludes patients who died in hospital.

**Args:**
- `conn`: DuckDB connection with `admissions` table loaded
- `windows`: List of readmission windows in days (default: `[30, 60]`)

#### `select_neurology_cohort()`

```python
def select_neurology_cohort(
    conn: duckdb.DuckDBPyConnection,
    icd_prefixes: list[str],
) -> pd.DataFrame
```

Select cohort based on ICD diagnosis codes and inclusion criteria.

**Inclusion criteria:**
- Age 18-89
- ICU stay > 24 hours and < 100 days (2400 hours)
- First ICU stay per admission
- Has diagnosis matching one of the ICD prefixes

**Args:**
- `conn`: DuckDB connection with required tables loaded
- `icd_prefixes`: List of ICD-10 code prefixes (e.g., `["I63", "I61"]`)

**Returns:**
DataFrame with columns: `subject_id`, `hadm_id`, `stay_id`

---

## Layer 2: Graph Construction

### `src.graph_construction.pipeline`

Graph construction pipeline orchestrator.

#### `build_graph()`

```python
def build_graph(
    db_path: Path,
    ontology_dir: Path,
    output_path: Path,
    icd_prefixes: list[str],
    patients_limit: int = 0,
    biomarkers_limit: int = 0,
    vitals_limit: int = 0,
    microbiology_limit: int = 0,
    prescriptions_limit: int = 0,
    diagnoses_limit: int = 0,
    skip_allen_relations: bool = False,
    snomed_mappings_dir: Path | None = None,
    umls_api_key: str | None = None,
) -> Graph
```

Build RDF graph from MIMIC-IV DuckDB database.

**Args:**
- `db_path`: Path to DuckDB database file
- `ontology_dir`: Path to directory containing ontology files
- `output_path`: Path to write RDF/XML output file
- `icd_prefixes`: List of ICD-10 prefixes to filter cohort
- `patients_limit`: Maximum patients to process (0 = no limit)
- `biomarkers_limit`: Maximum biomarker events per ICU stay (0 = no limit)
- `vitals_limit`: Maximum vital events per ICU stay (0 = no limit)
- `microbiology_limit`: Maximum microbiology events per ICU stay (0 = no limit)
- `prescriptions_limit`: Maximum prescriptions per admission (0 = no limit)
- `diagnoses_limit`: Maximum diagnoses per admission (0 = no limit)
- `skip_allen_relations`: If True, skip Allen relation computation
- `snomed_mappings_dir`: Path to directory containing SNOMED mapping JSON files
- `umls_api_key`: Optional UMLS API key for LOINC→SNOMED crosswalk. When set, unmapped LOINC codes are resolved via the UMLS REST API on first build and cached to `loinc_crosswalk_cache.json`; subsequent builds load from cache (zero API calls).

**Returns:**
The constructed RDF graph.

---

### `src.graph_construction.ontology`

Ontology initialization and namespace definitions.

#### Constants

```python
MIMIC_NS = Namespace("http://www.cnam.fr/MIMIC4-ICU-BSI/V1#")
TIME_NS = Namespace("http://www.w3.org/2006/time#")
```

#### `initialize_graph()`

```python
def initialize_graph(ontology_dir: Path) -> Graph
```

Initialize an RDF graph with loaded ontologies and bound namespaces.

**Args:**
- `ontology_dir`: Path to ontology definition files

**Returns:**
rdflib Graph with namespaces bound.

---

### `src.graph_construction.patient_writer`

Patient and admission RDF writer.

#### `write_patient()`

```python
def write_patient(graph: Graph, patient_data: dict) -> URIRef
```

Create Patient RDF node.

**Args:**
- `graph`: RDF graph to write to
- `patient_data`: Dict with `subject_id`, `gender`, `anchor_age`

**Returns:**
URIRef for the created Patient node.

#### `write_admission()`

```python
def write_admission(
    graph: Graph,
    admission_data: dict,
    patient_uri: URIRef
) -> URIRef
```

Create HospitalAdmission RDF node with temporal bounds.

**Args:**
- `graph`: RDF graph to write to
- `admission_data`: Dict with admission properties
- `patient_uri`: URIRef of the patient

**Returns:**
URIRef for the created HospitalAdmission node.

#### `link_sequential_admissions()`

```python
def link_sequential_admissions(
    graph: Graph,
    admission_uris: list[URIRef]
) -> None
```

Link sequential admissions with `followedBy` predicate.

**Args:**
- `graph`: RDF graph
- `admission_uris`: List of admission URIs in chronological order

---

### `src.graph_construction.event_writers`

Clinical event RDF writers.

#### `write_icu_stay()`

```python
def write_icu_stay(
    graph: Graph,
    stay_data: dict,
    admission_uri: URIRef
) -> URIRef
```

Create ICUStay interval node.

#### `write_icu_days()`

```python
def write_icu_days(
    graph: Graph,
    stay_data: dict,
    icu_stay_uri: URIRef
) -> list[dict]
```

Create ICUDay intervals for an ICU stay.

**Returns:**
List of ICU day metadata dicts with `uri`, `start`, `end`.

#### `write_biomarker_event()`

```python
def write_biomarker_event(
    graph: Graph,
    event_data: dict,
    icu_stay_uri: URIRef,
    icu_day_metadata: list[dict]
) -> URIRef
```

Create BioMarkerEvent node.

#### `write_clinical_sign_event()`

```python
def write_clinical_sign_event(
    graph: Graph,
    event_data: dict,
    icu_stay_uri: URIRef,
    icu_day_metadata: list[dict]
) -> URIRef
```

Create ClinicalSignEvent node (vital signs).

#### `write_microbiology_event()`

```python
def write_microbiology_event(
    graph: Graph,
    event_data: dict,
    icu_stay_uri: URIRef,
    icu_day_metadata: list[dict]
) -> URIRef
```

Create MicrobiologyEvent node.

#### `write_prescription_event()`

```python
def write_prescription_event(
    graph: Graph,
    event_data: dict,
    icu_stay_uri: URIRef,
    icu_day_metadata: list[dict]
) -> URIRef
```

Create PrescriptionEvent node.

#### `write_diagnosis_event()`

```python
def write_diagnosis_event(
    graph: Graph,
    event_data: dict,
    admission_uri: URIRef
) -> URIRef
```

Create DiagnosisEvent node.

---

### `src.graph_construction.temporal.allen_relations`

Allen temporal relation computation.

#### `compute_allen_relations()`

```python
def compute_allen_relations(
    graph: Graph,
    icu_stay_uri: URIRef
) -> int
```

Compute Allen temporal relations for events in a single ICU stay.

**Args:**
- `graph`: RDF graph containing the events
- `icu_stay_uri`: URI of the ICU stay to process

**Returns:**
Count of relation triples added.

#### `compute_allen_relations_for_patient()`

```python
def compute_allen_relations_for_patient(
    graph: Graph,
    patient_uri: URIRef
) -> int
```

Compute Allen relations for all ICU stays of a patient.

**Args:**
- `graph`: RDF graph containing the patient data
- `patient_uri`: URI of the patient

**Returns:**
Total count of relation triples added across all ICU stays.

---

### `src.graph_construction.terminology.snomed_mapper`

SNOMED-CT concept mapper.

#### `SnomedMapper`

```python
class SnomedMapper:
    def __init__(
        self,
        mappings_dir: Path,
        umls_api_key: str | None = None,
    ) -> None: ...
```

Resolve MIMIC-IV identifiers to SNOMED-CT concepts. Loads JSON mapping files lazily.

**Args:**
- `mappings_dir`: Path to directory containing `*_to_snomed.json` files
- `umls_api_key`: Optional UMLS API key. When set, `ensure_loinc_coverage()` can query unmapped LOINC codes via the UMLS crosswalk API.

#### `ensure_loinc_coverage()`

```python
def ensure_loinc_coverage(self, loinc_codes: list[str]) -> int
```

Query the UMLS crosswalk for any LOINC codes not already in the static mapping or cache. Results are merged into the in-memory map and persisted to `loinc_crosswalk_cache.json`.

**Args:**
- `loinc_codes`: List of LOINC codes to check

**Returns:**
Number of newly resolved codes (0 if no API key or all codes already mapped).

---

### `src.graph_construction.terminology.mapping_sources`

Pluggable LOINC→SNOMED mapping sources.

#### `StaticMappingSource`

```python
class StaticMappingSource:
    def __init__(self, json_path: Path) -> None: ...
    def lookup(self, loinc_code: str) -> dict | None: ...
    def lookup_batch(self, codes: list[str]) -> dict[str, dict]: ...
```

Reads from a pre-generated JSON file. Entries whose `snomed_code` does not match the 5-18 digit SCTID pattern are rejected.

#### `UMLSCrosswalkSource`

```python
class UMLSCrosswalkSource:
    def __init__(self, api_key: str, cache_path: Path) -> None: ...
    def lookup(self, loinc_code: str) -> dict | None: ...
    def lookup_batch(self, codes: list[str]) -> dict[str, dict]: ...
```

Queries the NLM UMLS REST crosswalk endpoint (LNC→SNOMEDCT_US) with a lazy disk cache. Cache hits skip the API; after `lookup_batch()` the cache is persisted to disk.

---

### `src.graph_construction.terminology.mapping_chain`

#### `MappingChain`

```python
class MappingChain:
    def __init__(self, sources: list[MappingSource]) -> None: ...
    def resolve(self, loinc_code: str) -> dict | None: ...
    def resolve_batch(self, codes: list[str]) -> dict[str, dict]: ...
```

Ordered waterfall of mapping sources. Tries each source in order; first hit wins. For batch resolution, each successive source only receives codes still unresolved.

---

## Layer 3: Graph Analysis

### `src.graph_analysis.analysis`

Graph metrics and analysis.

#### `count_nodes_by_type()`

```python
def count_nodes_by_type(rdf_graph: Graph) -> dict[str, int]
```

Count nodes by RDF type via SPARQL.

**Returns:**
Dict mapping type name to count.

#### `count_edges_by_type()`

```python
def count_edges_by_type(rdf_graph: Graph) -> dict[str, int]
```

Count edges by predicate via SPARQL.

**Returns:**
Dict mapping predicate name to count.

#### `generate_analysis_report()`

```python
def generate_analysis_report(
    graph: Graph,
    output_path: Path
) -> None
```

Generate markdown analysis report.

---

### `src.graph_analysis.rdf_to_networkx`

RDF to NetworkX conversion.

#### `rdf_to_networkx()`

```python
def rdf_to_networkx(
    rdf_graph: Graph,
    include_literals: bool = False
) -> nx.DiGraph
```

Convert RDF graph to NetworkX directed graph.

**Args:**
- `rdf_graph`: rdflib Graph
- `include_literals`: Whether to include literal nodes

**Returns:**
NetworkX DiGraph.

---

## Layer 4: Feature Extraction

### `src.feature_extraction.feature_builder`

Feature matrix builder.

#### `build_feature_matrix()`

```python
def build_feature_matrix(
    graph: Graph,
    save_path: Path | None = None
) -> pd.DataFrame
```

Build complete feature matrix from RDF graph.

Combines all feature extractors and readmission labels.

**Args:**
- `graph`: RDF graph containing the clinical knowledge graph
- `save_path`: Optional path to save as parquet

**Returns:**
DataFrame with all features and labels, one row per admission.

---

### `src.feature_extraction.tabular_features`

Tabular feature extraction from RDF graphs.

#### `extract_demographics()`

```python
def extract_demographics(graph: Graph) -> pd.DataFrame
```

Extract demographic features.

**Returns:**
DataFrame with columns: `hadm_id`, `age`, `gender_M`, `gender_F`

#### `extract_stay_features()`

```python
def extract_stay_features(graph: Graph) -> pd.DataFrame
```

Extract ICU stay features.

**Returns:**
DataFrame with columns: `hadm_id`, `icu_los_hours`, `num_icu_days`, `admission_type_*`

#### `extract_lab_summary()`

```python
def extract_lab_summary(graph: Graph) -> pd.DataFrame
```

Extract lab (biomarker) summary statistics.

**Returns:**
DataFrame with columns: `hadm_id`, `{biomarker}_mean/min/max/std/count/first/last/abnormal_rate`

#### `extract_vital_summary()`

```python
def extract_vital_summary(graph: Graph) -> pd.DataFrame
```

Extract vital sign summary statistics.

**Returns:**
DataFrame with columns: `hadm_id`, `{vital}_mean/min/max/std/cv/count`

#### `extract_medication_features()`

```python
def extract_medication_features(graph: Graph) -> pd.DataFrame
```

Extract medication features.

**Returns:**
DataFrame with columns: `hadm_id`, `num_distinct_meds`, `total_prescription_days`, `has_prescription`

#### `extract_diagnosis_features()`

```python
def extract_diagnosis_features(graph: Graph) -> pd.DataFrame
```

Extract diagnosis features.

**Returns:**
DataFrame with columns: `hadm_id`, `num_diagnoses`, `icd_chapter_*`

---

### `src.feature_extraction.graph_features`

Graph structure feature extraction.

#### `extract_temporal_features()`

```python
def extract_temporal_features(graph: Graph) -> pd.DataFrame
```

Extract temporal relation features.

**Returns:**
DataFrame with columns: `hadm_id`, `events_per_icu_day`, `num_before_relations`, `num_during_relations`, `total_temporal_edges`

#### `extract_graph_structure_features()`

```python
def extract_graph_structure_features(graph: Graph) -> pd.DataFrame
```

Extract graph structure features.

**Returns:**
DataFrame with columns: `hadm_id`, `patient_subgraph_nodes`, `patient_subgraph_edges`, `patient_subgraph_density`, `mean_node_degree`, `max_node_degree`

---

## Layer 5: Prediction

### `src.prediction.split`

Patient-level data splitting.

#### `patient_level_split()`

```python
def patient_level_split(
    df: pd.DataFrame,
    target_col: str,
    subject_col: str = "subject_id",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
```

Split data at patient level to prevent data leakage.

All admissions for a patient are placed in the same split.

**Args:**
- `df`: DataFrame with features, target, and subject_id columns
- `target_col`: Name of the target column
- `subject_col`: Name of the patient identifier column
- `test_size`: Fraction of patients for test set
- `val_size`: Fraction of patients for validation set
- `random_state`: Random seed for reproducibility

**Returns:**
Tuple of (train_df, val_df, test_df) DataFrames.

---

### `src.prediction.model`

Model training and persistence.

#### `train_model()`

```python
def train_model(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    model_type: Literal["logistic_regression", "xgboost"] = "xgboost",
    **kwargs,
) -> LogisticRegression | XGBClassifier
```

Train a classification model for readmission prediction.

**Args:**
- `X_train`: Training features
- `y_train`: Training labels
- `model_type`: `"logistic_regression"` or `"xgboost"`
- `**kwargs`: Additional parameters for the model constructor

**Returns:**
Fitted model.

#### `save_model()`

```python
def save_model(
    model: LogisticRegression | XGBClassifier,
    path: Path
) -> None
```

Save a trained model to disk.

XGBoost: JSON format. sklearn: Pickle format.

#### `load_model()`

```python
def load_model(path: Path) -> LogisticRegression | XGBClassifier
```

Load a trained model from disk.

---

### `src.prediction.evaluate`

Model evaluation and reporting.

#### `evaluate_model()`

```python
def evaluate_model(
    model: LogisticRegression | XGBClassifier,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> dict
```

Evaluate a trained model and compute performance metrics.

**Returns:**
Dict with `auroc`, `auprc`, `precision`, `recall`, `f1`, `threshold`, `confusion_matrix`.

#### `get_feature_importance()`

```python
def get_feature_importance(
    model: LogisticRegression | XGBClassifier,
    feature_names: list[str],
) -> pd.DataFrame
```

Extract feature importance from a trained model.

**Returns:**
DataFrame with columns `[feature, importance]`, sorted descending.

#### `calibration_data()`

```python
def calibration_data(
    model: LogisticRegression | XGBClassifier,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]
```

Compute calibration curve data.

**Returns:**
Tuple of `(fraction_of_positives, mean_predicted_value)`.

#### `generate_evaluation_report()`

```python
def generate_evaluation_report(
    metrics: dict,
    feature_importance: pd.DataFrame,
    output_path: Path,
) -> None
```

Generate a markdown evaluation report.

---

## Layer 5b: GNN Prediction Pathway

### `src.gnn.embeddings`

SapBERT-based concept embeddings for GNN node features. See [CITATIONS.md](../src/gnn/CITATIONS.md) for academic references.

#### `SapBERTEmbedder`

```python
class SapBERTEmbedder:
    MODEL_ID = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    EMBEDDING_DIM = 768

    def __init__(self, cache_dir: Path | None = None) -> None: ...
```

Generate dense 768-dim embeddings for clinical terms using SapBERT (Liu et al., NAACL 2021). The model is loaded lazily on first use.

**Args:**
- `cache_dir`: Optional HuggingFace cache directory for model weights

#### `SapBERTEmbedder.embed_terms()`

```python
def embed_terms(self, terms: list[str]) -> torch.Tensor
```

Embed clinical terms into 768-dim vectors using CLS pooling. Processes in batches of 64.

**Args:**
- `terms`: Clinical term strings to embed

**Returns:**
Tensor of shape `(len(terms), 768)`.

#### `SapBERTEmbedder.embed_and_cache()`

```python
def embed_and_cache(
    self,
    terms: list[str],
    labels: list[str],
    cache_path: Path,
) -> dict[str, torch.Tensor]
```

Embed terms and cache the result to disk. If `cache_path` exists, loads from cache without touching the model.

**Args:**
- `terms`: Clinical term strings to embed
- `labels`: Keys for the returned dict (one per term)
- `cache_path`: File path for the `.pt` cache

**Returns:**
Dict mapping each label to its 768-dim embedding tensor.

#### `build_concept_embeddings()`

```python
def build_concept_embeddings(
    snomed_mapper: SnomedMapper,
    cache_path: Path = Path("data/processed/concept_embeddings.pt"),
) -> dict[str, torch.Tensor]
```

Embed all unique SNOMED concepts from the mapper's 7 internal mapping dictionaries and cache to disk.

**Args:**
- `snomed_mapper`: A loaded SnomedMapper instance
- `cache_path`: Where to save/load the `.pt` cache

**Returns:**
Dict mapping SNOMED code to 768-dim embedding tensor.

#### `embed_unmapped_terms()`

```python
def embed_unmapped_terms(
    terms: list[str],
    cache_path: Path = Path("data/processed/unmapped_embeddings.pt"),
) -> dict[str, torch.Tensor]
```

Embed free-text terms that lack SNOMED mappings.

**Args:**
- `terms`: Raw clinical term strings
- `cache_path`: Where to save/load the `.pt` cache

**Returns:**
Dict mapping term string to 768-dim embedding tensor.

---

### `src.gnn.model`

TD4DD model assembly — combines the dual-track Transformer and cross-view diffusion branches into a single end-to-end model.

#### `ModelConfig`

```python
@dataclass
class ModelConfig:
    feat_dims: dict[str, int]       # node_type → raw feature dim
    d_model: int = 128
    num_contextual_paths: int = 2
    k_hops_contextual: int = 4
    k_hops_temporal: int = 2
    nhead: int = 4
    dropout: float = 0.3
    use_transformer: bool = True     # ablation toggle
    use_diffusion: bool = True       # ablation toggle
    use_temporal_encoding: bool = True
    diffusion_T: int = 100
    diffusion_ddim_steps: int = 10
```

Configuration dataclass for the TD4DD model. `use_transformer` and `use_diffusion` are ablation toggles that disable the respective branches.

#### `TD4DDModel`

```python
class TD4DDModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None: ...
```

Full TD4DD model with five components: per-type linear projections, optional `DualTrackTransformer`, optional `DiffusionModule`, a learnable fusion weight (`lambda_param`), and a binary classifier head.

#### `TD4DDModel.forward()`

```python
def forward(
    self,
    batch_data: HeteroData,
    aux_edge_indices: list[Tensor] | None = None,
    *,
    contextual_hop_neighbors: list[list[Tensor]] | None = None,
    temporal_hop_neighbors: list[Tensor] | None = None,
    temporal_hop_deltas: list[Tensor] | None = None,
) -> dict
```

Run the full model: project node features, run transformer and/or diffusion branches, fuse with learnable lambda, and classify.

**Args:**
- `batch_data`: PyG `HeteroData` with node features (must include `"admission"` node type)
- `aux_edge_indices`: Two auxiliary edge-index tensors for the diffusion branch
- `contextual_hop_neighbors`: Pre-structured neighbor tensors for the transformer contextual track
- `temporal_hop_neighbors`: Pre-structured neighbor tensors for the transformer temporal track
- `temporal_hop_deltas`: Day-gap tensors for temporal encoding

**Returns:**
Dict with keys: `logits` `(B, 1)`, `probabilities` `(B, 1)`, `h_hhgat`, `h_diff`, `L_diff`, `attention_info`.

---

### `src.gnn.losses`

Modular loss functions for imbalanced binary classification.

#### `LossConfig`

```python
@dataclass
class LossConfig:
    cls_loss_type: str = "bce"          # "bce" | "focal" | "bce_smoothed"
    pos_weight: float | None = None     # auto-compute from labels if None
    label_smoothing: float = 0.05
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    diff_weight: float = 1.0
```

#### `sigmoid_focal_loss()`

```python
def sigmoid_focal_loss(
    pred: Tensor, target: Tensor,
    alpha: float = 0.25, gamma: float = 2.0,
) -> Tensor
```

Focal loss for imbalanced classification. Uses `torchvision.ops.sigmoid_focal_loss` when available, otherwise a manual implementation.

#### `build_classification_loss()`

```python
def build_classification_loss(
    config: LossConfig, labels: Tensor,
) -> Callable[[Tensor, Tensor], Tensor]
```

Factory that returns a loss callable based on `config.cls_loss_type`. Auto-computes `pos_weight = n_neg / n_pos` when `config.pos_weight is None`.

#### `compute_total_loss()`

```python
def compute_total_loss(
    cls_loss_fn: Callable[[Tensor, Tensor], Tensor],
    logits: Tensor, labels: Tensor,
    L_diff: Tensor, config: LossConfig,
) -> dict[str, Tensor]
```

Combine classification and diffusion losses: `total = cls + diff_weight * L_diff`.

**Returns:**
Dict with keys: `total`, `cls`, `diff`.

---

## Configuration

### `config.settings`

Pydantic configuration module.

#### `Settings`

```python
class Settings(BaseSettings):
    # Paths
    mimic_iv_path: Path
    duckdb_path: Path
    clinical_tkg_repo: Path

    # Neo4j
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # Data source
    data_source: Literal["local", "bigquery"]
    bigquery_project: str | None

    # SNOMED-CT mappings
    snomed_mappings_dir: Path | None

    # UMLS API
    umls_api_key: str | None  # Optional: enables LOINC→SNOMED crosswalk

    # Cohort configuration
    cohort_icd_codes: list[str]
    readmission_window_days: int
    patients_limit: int
    biomarkers_limit: int
    vitals_limit: int
    diagnoses_limit: int
    skip_allen_relations: bool
```

Configuration loaded from environment variables and `.env` file.

**Example:**
```python
from config.settings import Settings

settings = Settings()
print(settings.mimic_iv_path)

# Override settings
new_settings = settings.model_copy(update={"patients_limit": 100})
```
