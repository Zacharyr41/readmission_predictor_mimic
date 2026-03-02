"""WLST experiment registry and runner.

Defines W1-W7 experiment configurations for Stage 1 (clinical trajectory)
and Stage 2 (non-clinical confounders) ablation studies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WLSTExperimentConfig:
    """Configuration for a single WLST experiment."""

    name: str = ""
    description: str = ""
    stage: str = "stage1"

    # Model type
    model_type: str = "gnn"  # "gnn", "logistic_regression", "xgboost", "confounders_only"

    # GNN toggles (only used when model_type == "gnn")
    use_transformer: bool = True
    use_diffusion: bool = True
    use_temporal_encoding: bool = True

    # Architecture params
    d_model: int = 128
    nhead: int = 4
    dropout: float = 0.3
    lr: float = 0.001
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 20
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: _convert(v) for k, v in d.items()}


def _convert(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


# ── Stage 1 experiments: clinical trajectory only ──

WLST_EXPERIMENT_REGISTRY: dict[str, WLSTExperimentConfig] = {
    "W1_mlp_baseline": WLSTExperimentConfig(
        name="W1_mlp_baseline",
        description="Floor: projection + classifier only (no graph structure)",
        stage="stage1",
        use_transformer=False,
        use_diffusion=False,
        use_temporal_encoding=False,
    ),
    "W2_transformer_only": WLSTExperimentConfig(
        name="W2_transformer_only",
        description="Graph structure contribution (transformer, no diffusion/temporal)",
        stage="stage1",
        use_transformer=True,
        use_diffusion=False,
        use_temporal_encoding=False,
    ),
    "W3_transformer_temporal": WLSTExperimentConfig(
        name="W3_transformer_temporal",
        description="Temporal encoding contribution (48h trajectory)",
        stage="stage1",
        use_transformer=True,
        use_diffusion=False,
        use_temporal_encoding=True,
    ),
    "W4_full_model": WLSTExperimentConfig(
        name="W4_full_model",
        description="Full TD4DD on Stage 1 graph",
        stage="stage1",
        use_transformer=True,
        use_diffusion=True,
        use_temporal_encoding=True,
    ),

    # ── Stage 2 experiments: non-clinical confounders ──

    "W5_stage2_tabular": WLSTExperimentConfig(
        name="W5_stage2_tabular",
        description="Tabular model with all features (clinical + confounders)",
        stage="stage2",
        model_type="xgboost",
    ),
    "W6_stage2_full_model": WLSTExperimentConfig(
        name="W6_stage2_full_model",
        description="Full TD4DD on Stage 2 extended graph",
        stage="stage2",
        use_transformer=True,
        use_diffusion=True,
        use_temporal_encoding=True,
    ),
    "W7_confounders_only": WLSTExperimentConfig(
        name="W7_confounders_only",
        description="Only non-clinical features (ablation: confounders alone)",
        stage="stage2",
        model_type="confounders_only",
    ),
}


def get_wlst_gnn_registry() -> dict:
    """Convert WLST GNN experiments to ExperimentConfig objects for ExperimentRunner.

    Only includes experiments with model_type == "gnn". Non-GNN experiments
    (xgboost, confounders_only) are handled by run_classical_baselines.
    """
    from src.gnn.experiments import ExperimentConfig
    from src.gnn.losses import LossConfig
    from src.gnn.train import TrainingConfig
    from src.gnn.view_adapter import GraphViewConfig

    registry = {}
    for name, wlst_cfg in WLST_EXPERIMENT_REGISTRY.items():
        if wlst_cfg.model_type != "gnn":
            continue
        registry[name] = ExperimentConfig(
            name=name,
            description=wlst_cfg.description,
            use_transformer=wlst_cfg.use_transformer,
            use_diffusion=wlst_cfg.use_diffusion,
            use_temporal_encoding=wlst_cfg.use_temporal_encoding,
            d_model=wlst_cfg.d_model,
            nhead=wlst_cfg.nhead,
            dropout=wlst_cfg.dropout,
            view_config=GraphViewConfig.wlst_default(),
            loss_config=LossConfig(),
            training_config=TrainingConfig(
                lr=wlst_cfg.lr,
                batch_size=wlst_cfg.batch_size,
                max_epochs=wlst_cfg.max_epochs,
                patience=wlst_cfg.patience,
                weight_decay=wlst_cfg.weight_decay,
                grad_clip_norm=wlst_cfg.grad_clip_norm,
            ),
        )
    return registry


# Classical baselines (run alongside GNN experiments)
CLASSICAL_BASELINES = {
    "logistic_regression": WLSTExperimentConfig(
        name="logistic_regression",
        description="Logistic Regression with L1 penalty + balanced class weights",
        model_type="logistic_regression",
    ),
    "xgboost": WLSTExperimentConfig(
        name="xgboost",
        description="XGBoost with scale_pos_weight",
        model_type="xgboost",
    ),
}


def run_classical_baselines(
    feature_df: pd.DataFrame,
    label_col: str = "wlst_label",
    output_dir: Path | None = None,
    seed: int = 42,
) -> dict[str, dict]:
    """Run classical ML baselines on tabular features.

    Args:
        feature_df: DataFrame with features and label column.
        label_col: Name of the label column.
        output_dir: Directory to save model artifacts.
        seed: Random seed.

    Returns:
        Dict mapping model name to evaluation metrics.
    """
    from src.prediction.evaluate import get_feature_importance
    from src.prediction.model import save_model, train_model
    from src.prediction.split import patient_level_split
    from src.wlst.evaluate import compute_wlst_metrics

    # Identify feature columns (exclude identifiers and labels)
    id_cols = {"subject_id", "hadm_id", "stay_id", "wlst_label", "outcome_category"}
    non_numeric = set()
    for col in feature_df.columns:
        if feature_df[col].dtype == "object":
            non_numeric.add(col)

    feature_cols = [c for c in feature_df.columns if c not in id_cols and c not in non_numeric]

    # Fill NaN with 0 for ML baselines
    feature_df_clean = feature_df.copy()
    feature_df_clean[feature_cols] = feature_df_clean[feature_cols].fillna(0)

    # Patient-level split (70/15/15)
    train_df, val_df, test_df = patient_level_split(
        feature_df_clean,
        target_col=label_col,
        test_size=0.15,
        val_size=0.15,
        random_state=seed,
    )
    X_train = train_df[feature_cols]
    y_train = train_df[label_col]
    X_test = test_df[feature_cols]
    y_test = test_df[label_col]

    results = {}
    for model_name, config in CLASSICAL_BASELINES.items():
        logger.info(f"Training {model_name}...")
        model = train_model(
            X_train, y_train,
            model_type=config.model_type,
            random_state=seed,
        )

        y_proba = model.predict_proba(X_test)[:, 1]
        outcome_cats = test_df["outcome_category"] if "outcome_category" in test_df.columns else None
        metrics = compute_wlst_metrics(y_proba, np.asarray(y_test), outcome_categories=outcome_cats)
        importance = get_feature_importance(model, feature_cols)

        results[model_name] = {
            "metrics": metrics,
            "feature_importance": importance,
            "config": config.to_dict(),
        }

        logger.info(f"  {model_name}: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}")

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            ext = ".json" if model_name == "xgboost" else ".pkl"
            save_model(model, output_dir / f"{model_name}{ext}")

    return results
