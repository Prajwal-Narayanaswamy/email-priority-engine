"""CLI: build the sender-frequency lookup from the training feature parquet."""

from __future__ import annotations

import logging

import click
import pandas as pd

from src.serving import config
from src.serving.lookup import build_sender_lookup, save_sender_lookup

logger = logging.getLogger(__name__)


@click.command()
@click.option("--features", type=click.Path(exists=True), default=config.FEATURES_PATH)
@click.option("--output", type=click.Path(), default=config.SENDER_LOOKUP_PATH)
def main(features: str, output: str) -> None:
    """Build and save the sender -> sender_frequency JSON lookup."""
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)
    df = pd.read_parquet(features, columns=["from", "sender_frequency"])
    save_sender_lookup(build_sender_lookup(df), Path(output))


if __name__ == "__main__":
    main()
