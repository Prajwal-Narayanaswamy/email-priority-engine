"""Register the ADR-013 reference model as `email-priority` with alias `champion`.

Dry run by default: prints the exact tags and actions and writes nothing. Pass
--apply to write. Documented as reference-only (ADR-013/ADR-014); the CI/CD stage
will promote by moving the alias.

Run from the repo root::

    python -m scripts.register_reference_model            # dry run
    python -m scripts.register_reference_model --apply    # write
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from mlflow import MlflowClient

from src.serving import config
from src.serving.registration import PrecheckError, apply_plan, build_plan
from src.serving.registry import (
    REFERENCE_EVAL_RUN_ID,
    REFERENCE_SOURCE_RUN_ID,
    configure_mlflow,
)

logger = logging.getLogger(__name__)


@click.command()
@click.option("--source-run-id", default=REFERENCE_SOURCE_RUN_ID, show_default=True)
@click.option("--eval-run-id", default=REFERENCE_EVAL_RUN_ID, show_default=True)
@click.option(
    "--lookup",
    "lookup_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=config.SENDER_LOOKUP_PATH,
    show_default=True,
)
@click.option(
    "--features",
    "features_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=config.FEATURES_PATH,
    show_default=True,
)
@click.option("--apply", "apply", is_flag=True, help="Write to the registry.")
def main(
    source_run_id: str,
    eval_run_id: str,
    lookup_path: Path,
    features_path: Path,
    apply: bool,
) -> None:
    """Run prechecks, print the plan, and write it only with --apply."""
    logging.basicConfig(level=logging.INFO)
    configure_mlflow(config.TRACKING_URI)
    logger.info("Tracking and registry URI: %s", config.TRACKING_URI)
    client = MlflowClient(tracking_uri=config.TRACKING_URI)
    try:
        plan = build_plan(
            client,
            source_run_id=source_run_id,
            eval_run_id=eval_run_id,
            lookup_path=lookup_path,
            features_path=features_path,
        )
    except PrecheckError as exc:
        click.echo(f"PRECHECK FAILED: {exc}", err=True)
        sys.exit(1)
    click.echo("Prechecks passed. Plan:")
    for line in plan.describe():
        click.echo(f"  {line}")
    if not apply:
        click.echo("Dry run: nothing written. Re-run with --apply to write.")
        return
    version = apply_plan(client, plan)
    click.echo(f"Registered {plan.model_name} version {version} alias {plan.alias}.")


if __name__ == "__main__":
    main()
