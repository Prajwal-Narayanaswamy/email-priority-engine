"""Prefect task wrappers around the pure ingestion functions."""

from __future__ import annotations

from pathlib import Path

from prefect import get_run_logger, task

from src.ingestion.ingest import DEFAULT_BATCH_SIZE, ingest


@task
def ingest_emails_task(
    archive: Path,
    output_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
) -> Path:
    """Stream-extract, validate, label, and persist Enron emails."""
    logger = get_run_logger()
    logger.info(
        "Starting streaming ingestion from %s (batch_size=%d)", archive, batch_size
    )
    written = ingest(
        archive, output_path, limit=limit, batch_size=batch_size, logger=logger
    )
    logger.info("Ingestion complete: %s", written)
    return written
