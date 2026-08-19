"""Prefect flow for feature engineering."""

from __future__ import annotations

from pathlib import Path

import click
from prefect import flow, get_run_logger

from src.tasks.feature_tasks import feature_engineering_task


@flow(name="feature_flow")
def feature_flow(
    input_path: str,
    output_path: str = "data/features/features.parquet",
    limit: int | None = None,
) -> Path:
    """Engineer tabular features from the labeled email parquet."""
    logger = get_run_logger()
    logger.info("Starting feature_flow for %s", input_path)
    return feature_engineering_task(Path(input_path), Path(output_path), limit=limit)


@click.command()
@click.option(
    "--input",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="data/processed/emails.parquet",
    help="Input labeled Parquet path.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="data/features/features.parquet",
    help="Output features Parquet path.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=None,
    help="Process only the first N rows (optional).",
)
def main(input: Path, output: Path, limit: int | None) -> None:
    """CLI entry point: run the feature engineering flow."""
    feature_flow(input_path=str(input), output_path=str(output), limit=limit)


if __name__ == "__main__":
    main()
