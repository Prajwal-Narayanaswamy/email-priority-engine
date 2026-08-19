"""Prefect flow for email ingestion."""

from __future__ import annotations

from pathlib import Path

import click
from prefect import flow, get_run_logger

from src.ingestion.ingest import DEFAULT_BATCH_SIZE
from src.tasks.ingest_tasks import ingest_emails_task


@flow(name="ingest_flow")
def ingest_flow(
    tar_path: str,
    output_path: str = "data/processed/emails.parquet",
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Path:
    """Stream-extract, validate, label, and persist Enron emails."""
    logger = get_run_logger()
    logger.info("Starting ingest_flow for %s", tar_path)
    return ingest_emails_task(
        Path(tar_path), Path(output_path), batch_size=batch_size, limit=limit
    )


@click.command()
@click.option(
    "--tar",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="data/raw/enron_mail_20150507.tar.gz",
    help="Path to the Enron tar.gz archive.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="data/processed/emails.parquet",
    help="Output Parquet path.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=DEFAULT_BATCH_SIZE,
    show_default=True,
    help="Number of emails to buffer before each Parquet write.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=None,
    help="Process only the first N emails (optional).",
)
def main(tar: Path, output: Path, batch_size: int, limit: int | None) -> None:
    """CLI entry point: run the ingestion flow."""
    ingest_flow(
        tar_path=str(tar), output_path=str(output), limit=limit, batch_size=batch_size
    )


if __name__ == "__main__":
    main()
