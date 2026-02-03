"""Patient-level data splitting for hospital readmission prediction.

Ensures no data leakage by keeping all admissions for a patient in the same split.
Uses stratified sampling to preserve class balance across train/val/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def patient_level_split(
    df: pd.DataFrame,
    target_col: str,
    subject_col: str = "subject_id",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data at patient level to prevent data leakage.

    All admissions for a patient are placed in the same split. Stratification
    is performed on patient-level labels using an "any-positive" strategy:
    a patient is labeled positive if any of their admissions is positive.

    Args:
        df: DataFrame with features, target, and subject_id columns
        target_col: Name of the target column (e.g., "readmitted_30d")
        subject_col: Name of the patient identifier column
        test_size: Fraction of patients for test set (default 0.15)
        val_size: Fraction of patients for validation set (default 0.15)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames
    """
    # Compute patient-level labels using any-positive strategy
    patient_labels = df.groupby(subject_col)[target_col].max().reset_index()
    patient_labels.columns = [subject_col, "patient_label"]

    patients = patient_labels[subject_col].values
    labels = patient_labels["patient_label"].values

    # First split: train+val vs test
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_val_idx, test_idx = next(sss_test.split(patients, labels))

    train_val_patients = patients[train_val_idx]
    test_patients = patients[test_idx]

    # Second split: train vs val (adjust val_size relative to train+val)
    train_val_labels = labels[train_val_idx]
    relative_val_size = val_size / (1 - test_size)

    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=relative_val_size, random_state=random_state
    )
    train_idx_rel, val_idx_rel = next(sss_val.split(train_val_patients, train_val_labels))

    train_patients = train_val_patients[train_idx_rel]
    val_patients = train_val_patients[val_idx_rel]

    # Map patient splits back to admissions
    train_df = df[df[subject_col].isin(train_patients)].copy()
    val_df = df[df[subject_col].isin(val_patients)].copy()
    test_df = df[df[subject_col].isin(test_patients)].copy()

    return train_df, val_df, test_df
