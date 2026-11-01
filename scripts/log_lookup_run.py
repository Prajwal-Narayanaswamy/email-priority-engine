"""Log the sender lookup to a new run in the `email-priority-serving` experiment.

Never touches the training experiment or any finished run. Idempotent: if a
finished run already holds a lookup with the same sha256, it is reused.

Run from the repo root::

    python -m scripts.log_lookup_run
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
from mlflow import MlflowClient

from src.serving import config
from src.serving.registration import log_lookup_run
from src.serving.registry import REFERENCE_SOURCE_RUN_ID, configure_mlflow

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--lookup",
    "lookup_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=config.SENDER_LOOKUP_PATH,
    show_default=True,
)
@click.option("--source-run-id", default=REFERENCE_SOURCE_RUN_ID, show_default=True)
def main(lookup_path: Path, source_run_id: str) -> None:
    """Log the lookup file, its hashes and its row count to the serving experiment."""
    logging.basicConfig(level=logging.INFO)
    configure_mlflow(config.TRACKING_URI)
    logger.info("Tracking and registry URI: %s", config.TRACKING_URI)
    client = MlflowClient(tracking_uri=config.TRACKING_URI)
    run_id = log_lookup_run(client, lookup_path, source_run_id)
    click.echo(f"lookup run id: {run_id}")


if __name__ == "__main__":
    main()
