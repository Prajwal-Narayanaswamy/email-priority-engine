"""Prefect flow for evaluating the behavioral-only model.

Order (frozen): split rebuild -> model load -> main evaluation (gate, score,
tune on val, report on test) -> diagnostic retrain -> Evidently baseline.
Evidently runs last so its failure cannot lose the evaluation results.

There is deliberately NO --limit: the split gate compares against the
full-data training counts, so a subsample would abort the run by design.
"""

from __future__ import annotations

from dotenv import load_dotenv

# MUST run before `src.tasks.evaluate_tasks` is imported. That module (via
# train_tasks) reads MLFLOW_TRACKING_URI from os.environ at IMPORT time to
# build TRACKING_URI, so a load_dotenv() placed below these imports would come
# too late and silently have no effect. Everything below is a late import.
load_dotenv()

from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import click  # noqa: E402
from prefect import flow, get_run_logger  # noqa: E402

from src.tasks.evaluate_tasks import (  # noqa: E402
    DEFAULT_SOURCE_RUN_ID,
    REFERENCE_CAP,
    TRACKING_URI,
    evaluate_variant_task,
    evidently_baseline_task,
    load_model_task,
    load_split_task,
    retrain_diagnostic_task,
)
from src.training.train import SPLIT_SEED  # noqa: E402


@flow(name="evaluate_flow")
def evaluate_flow(
    input_path: str,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    seed: int = SPLIT_SEED,
    reference_cap: int = REFERENCE_CAP,
) -> dict[str, Any]:
    """Evaluate behavioral-only end to end and return the three summaries.

    Args:
        input_path: Feature parquet path (full data; no subsampling).
        source_run_id: Training run under evaluation (behavioral-only).
        seed: Split seed; must reproduce the training split exactly.
        reference_cap: Evidently train-reference row cap.

    Returns:
        Dict with the `evaluation`, `diagnostic`, and `drift` summaries.
    """
    logger = get_run_logger()
    logger.info("Starting evaluate_flow for %s", input_path)
    logger.info("MLflow tracking URI: %s", TRACKING_URI)
    logger.info("Source run: %s", source_run_id)

    splits = load_split_task(input_path=Path(input_path), seed=seed)
    model = load_model_task(source_run_id=source_run_id)
    evaluation = evaluate_variant_task(
        model,
        splits,
        source_run_id=source_run_id,
        reference_cap=reference_cap,
        seed=seed,
    )
    diagnostic = retrain_diagnostic_task(
        splits["train"],
        splits["val"],
        evaluation["run_id"],
        evaluation["val"]["pr_auc"],
        source_run_id=source_run_id,
        artifact_num_trees=evaluation["loaded_num_trees"],
        artifact_best_iteration=evaluation["source_best_iteration"],
    )
    drift = evidently_baseline_task(
        splits["train"],
        splits["test"],
        evaluation["run_id"],
        reference_cap=reference_cap,
        seed=seed,
    )

    logger.info(
        "Evaluation run %s: threshold %.4f, test F1 %.4f, test PR-AUC %.4f, "
        "drift share %.3f, diagnostic gap %.2e",
        evaluation["run_id"],
        evaluation["tuned_threshold"],
        evaluation["test"]["f1"],
        evaluation["test"]["pr_auc"],
        drift["drift_share"],
        diagnostic["sanity_gap"],
    )
    logger.info("evaluate_flow complete")
    return {"evaluation": evaluation, "diagnostic": diagnostic, "drift": drift}


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="data/features/features.parquet",
    help="Input features Parquet path (full data; no --limit exists).",
)
@click.option(
    "--run-id",
    "source_run_id",
    type=str,
    required=True,
    help="Source training run id under evaluation (behavioral-only).",
)
@click.option("--seed", type=int, default=SPLIT_SEED)
@click.option(
    "--reference-cap",
    type=click.IntRange(min=1),
    default=REFERENCE_CAP,
    help="Evidently train-reference row cap.",
)
def main(
    input_path: Path,
    source_run_id: str,
    seed: int,
    reference_cap: int,
) -> None:
    """CLI entry point: run the evaluation flow."""
    evaluate_flow(
        input_path=str(input_path),
        source_run_id=source_run_id,
        seed=seed,
        reference_cap=reference_cap,
    )


if __name__ == "__main__":
    main()
