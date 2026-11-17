"""CLI: build the sender-frequency lookup (v2) from the training feature parquet."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd

from src.serving import config
from src.serving.lookup import (
    build_header,
    build_sender_lookup,
    file_sha256,
    save_sender_lookup,
)

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--features",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=config.FEATURES_PATH,
)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=None)
def main(features: Path, output: Path | None) -> None:
    """Build the sender lookup with a header (parquet sha256, rows, build date)."""
    logging.basicConfig(level=logging.INFO)
    target = output or config.SENDER_LOOKUP_PATH
    parquet_hash = file_sha256(features)
    df = pd.read_parquet(features, columns=["from", "sender_frequency"])
    header = build_header(parquet_hash, len(df))
    lookup_hash = save_sender_lookup(build_sender_lookup(df), target, header)
    logger.info("parquet_sha256=%s rows=%d", parquet_hash, len(df))
    logger.info("lookup_sha256=%s file=%s", lookup_hash, target)


if __name__ == "__main__":
    main()
