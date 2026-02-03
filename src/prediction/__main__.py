"""CLI entry point for prediction module.

Usage:
    python -m src.prediction [--features PATH] [--output-dir PATH]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.prediction.split import patient_level_split
from src.prediction.model import train_model, save_model
from src.prediction.evaluate import (
    evaluate_model,
    get_feature_importance,
    generate_evaluation_report,
)


logger = logging.getLogger(__name__)


def main():
    """Train and evaluate readmission prediction models."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate readmission prediction models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--features",
        "-f",
        type=Path,
        default=Path("data/features/feature_matrix.parquet"),
        help="Path to feature matrix parquet file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("outputs"),
        help="Output directory for models and reports",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="readmitted_30d",
        choices=["readmitted_30d", "readmitted_60d"],
        help="Target variable to predict",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        nargs="+",
        default=["logistic_regression", "xgboost"],
        choices=["logistic_regression", "xgboost"],
        help="Model types to train",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Check input exists
    if not args.features.exists():
        logger.error(f"Feature file not found: {args.features}")
        return 1

    logger.info(f"Loading features from {args.features}")

    # Load features
    feature_df = pd.read_parquet(args.features)
    logger.info(f"  Shape: {feature_df.shape}")

    # Validate required columns
    if "subject_id" not in feature_df.columns:
        logger.error("subject_id column required for patient-level splitting")
        return 1

    if args.target not in feature_df.columns:
        logger.error(f"Target column {args.target} not found")
        return 1

    # Check for sufficient data
    n_positive = feature_df[args.target].sum()
    n_negative = len(feature_df) - n_positive

    logger.info(f"  Class distribution: {n_positive} positive, {n_negative} negative")

    if n_positive < 2 or n_negative < 2:
        logger.error("Insufficient samples for stratified split (need 2+ per class)")
        return 1

    # Identify feature columns
    exclude_cols = ["hadm_id", "subject_id", "readmitted_30d", "readmitted_60d"]
    feature_cols = [c for c in feature_df.columns if c not in exclude_cols]
    logger.info(f"  Using {len(feature_cols)} features")

    # Split data
    logger.info("Performing patient-level split...")
    try:
        train_df, val_df, test_df = patient_level_split(
            feature_df,
            target_col=args.target,
            subject_col="subject_id",
        )
    except ValueError as e:
        logger.error(f"Split failed: {e}")
        return 1

    logger.info(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    if len(test_df) == 0:
        logger.error("Empty test set - not enough data")
        return 1

    X_train = train_df[feature_cols]
    y_train = train_df[args.target]
    X_test = test_df[feature_cols]
    y_test = test_df[args.target]

    # Setup output directories
    models_dir = args.output_dir / "models"
    reports_dir = args.output_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Train and evaluate models
    results = {}

    for model_type in args.model_type:
        logger.info(f"Training {model_type}...")

        try:
            # Train
            model = train_model(X_train, y_train, model_type=model_type)

            # Save model
            if model_type == "xgboost":
                model_path = models_dir / "xgboost.json"
            else:
                model_path = models_dir / f"{model_type}.pkl"
            save_model(model, model_path)
            logger.info(f"  Model saved to {model_path}")

            # Evaluate
            metrics = evaluate_model(model, X_test, y_test)
            importance = get_feature_importance(model, feature_cols)

            # Generate report
            report_path = reports_dir / f"evaluation_{model_type.replace('_', '')}.md"
            generate_evaluation_report(metrics, importance, report_path)
            logger.info(f"  Report saved to {report_path}")

            results[model_type] = metrics
            logger.info(f"  AUROC: {metrics['auroc']:.4f}")

        except Exception as e:
            logger.warning(f"  {model_type} failed: {e}")

    # Print summary
    print("\n" + "=" * 50)
    print("Training Complete")
    print("=" * 50)

    for model_type, metrics in results.items():
        print(f"\n{model_type}:")
        print(f"  AUROC:     {metrics['auroc']:.4f}")
        print(f"  AUPRC:     {metrics['auprc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")

    return 0


if __name__ == "__main__":
    exit(main())
