"""Prefect flow for model training and MLflow experiment tracking."""

from __future__ import annotations

from dotenv import load_dotenv

# MUST run before `src.tasks.train_tasks` is imported. That module reads
# MLFLOW_TRACKING_URI from os.environ at IMPORT time to build TRACKING_URI, so a
# load_dotenv() placed below these imports would come too late and silently have
# no effect. Everything below this line is therefore a late import.
load_dotenv()

from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import click  # noqa: E402
from prefect import flow, get_run_logger  # noqa: E402

from src.tasks.train_tasks import (  # noqa: E402
    TRACKING_URI,
    compare_variants,
    load_features_task,
    train_variant_task,
)
from src.training.train import SPLIT_SEED, VARIANTS  # noqa: E402


@flow(name="train_flow")
def train_flow(
    input_path: str,
    limit: int | None = None,
    variants: list[str] | None = None,
    n_estimators: int = 2000,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_features: int = 100,
    seed: int = SPLIT_SEED,
    stopping_rounds: int = 100,
) -> list[dict[str, Any]]:
    """Train every requested variant, logging each as its own MLflow run.

    Both variants share one 70/15/15 stratified split (same seed), so the
    difference in score is attributable to the feature set and not to sampling
    luck. `with-keyword-features` is expected to score higher for the reason
    documented in `train.LIMITATION_NOTE` — label leakage, not better modelling.

    Args:
        input_path: Feature parquet path.
        limit: Train on a stratified sample of N rows (smoke tests only).
        variants: Variant names to run; defaults to all of them.
        n_estimators: LightGBM boosting rounds (upper bound).
        learning_rate: Boosting learning rate.
        num_leaves: LightGBM tree complexity.
        max_features: TF-IDF vocabulary size.
        seed: Split and model seed.
        stopping_rounds: Early-stopping patience.

    Returns:
        One summary dict per variant, from `train_variant_task`.
    """
    logger = get_run_logger()
    selected = variants or [v.name for v in VARIANTS]

    logger.info("Starting train_flow for %s", input_path)
    logger.info("MLflow tracking URI: %s", TRACKING_URI)
    logger.info("Variants to run: %s", ", ".join(selected))

    df = load_features_task(Path(input_path), limit=limit, seed=seed)

    results = []
    for name in selected:
        results.append(
            train_variant_task(
                df=df,
                variant_name=name,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                max_features=max_features,
                seed=seed,
                stopping_rounds=stopping_rounds,
            )
        )

    logger.info(
        "Variant comparison (primary metric: class-1 F1):\n%s",
        compare_variants(results),
    )
    logger.info("train_flow complete")
    return results


@click.command()
@click.option(
    "--input",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="data/features/features.parquet",
    help="Input features Parquet path.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=None,
    help="Train on a stratified sample of N rows (smoke tests only).",
)
@click.option(
    "--variant",
    "variants",
    type=click.Choice([v.name for v in VARIANTS]),
    multiple=True,
    default=None,
    help="Run only this variant; repeatable. Defaults to all.",
)
@click.option("--n-estimators", type=click.IntRange(min=1), default=2000)
@click.option("--learning-rate", type=click.FloatRange(min=1e-4, max=1.0), default=0.05)
@click.option("--num-leaves", type=click.IntRange(min=2), default=31)
@click.option("--max-features", type=click.IntRange(min=1), default=100)
@click.option("--seed", type=int, default=SPLIT_SEED)
@click.option("--stopping-rounds", type=click.IntRange(min=1), default=100)
def main(
    input: Path,
    limit: int | None,
    variants: tuple[str, ...],
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    max_features: int,
    seed: int,
    stopping_rounds: int,
) -> None:
    """CLI entry point: run the training flow."""
    train_flow(
        input_path=str(input),
        limit=limit,
        variants=list(variants) or None,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_features=max_features,
        seed=seed,
        stopping_rounds=stopping_rounds,
    )


if __name__ == "__main__":
    main()
