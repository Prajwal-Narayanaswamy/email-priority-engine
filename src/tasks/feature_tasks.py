"""Prefect task wrappers around the pure feature-engineering functions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from prefect import get_run_logger, task

from src.features.features import KEEP_COLUMNS, engineer_features


@task
def feature_engineering_task(
    input_path: Path,
    output_path: Path,
    limit: int | None = None,
) -> Path:
    """Read the labeled parquet, engineer tabular features, persist the result.

    Reads only the light columns (uid, from, subject, date, priority) so the
    511MB body column never enters memory (ADR-008). TF-IDF is intentionally
    not applied here — train_flow fits/transforms on train/test splits only.
    """
    logger = get_run_logger()
    logger.info("Reading %s (columns only)...", input_path)
    df = pd.read_parquet(input_path, columns=list(KEEP_COLUMNS))
    if limit is not None:
        df = df.head(limit)
    logger.info("Read %d rows; engineering features...", len(df))
    engineered = engineer_features(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engineered.to_parquet(output_path, index=False)
    logger.info("Wrote %d rows x %d cols to %s", *engineered.shape, output_path)
    return output_path
