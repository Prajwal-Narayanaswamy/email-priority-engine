"""Prefect task wrappers and MLflow logging around the pure training functions.

This is the only module that touches MLflow. Every run is wrapped in
`with mlflow.start_run():` — no model is ever logged outside a run context.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn  # noqa: F401  (registers mlflow.sklearn.log_model)
import pandas as pd
from prefect import get_run_logger, task

from src.features.features import FEATURE_COLUMNS, fit_subject_vectorizer
from src.training.train import (
    LIMITATION_NOTE,
    SCALE_POS_WEIGHT,
    SPLIT_SEED,
    SUBJECT,
    TARGET,
    build_matrices,
    build_model,
    evaluate,
    feature_importance_frame,
    plot_feature_importance,
    stratified_split,
    train_model,
    variant_by_name,
)

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_DEFAULT_URI: str = f"sqlite:///{_REPO_ROOT}/mlruns/mlflow.db"

EXPERIMENT_NAME: str = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME", "email-priority-classifier"
)
# Default backend is SQLite, not a file store. MLflow 3.x marks the filesystem
# store as "maintenance mode" and raises on any read or write unless
# MLFLOW_ALLOW_FILE_STORE=true is set; SQLite is MLflow's documented replacement
# and needs no opt-in. ADR-002 (local tracking, zero cost) is unchanged.
#
# The default is an ABSOLUTE path anchored to the repo root. A relative
# `sqlite:///mlruns/mlflow.db` resolves against the process CWD, so launching
# the flow from anywhere but the repo root silently created a second, invisible
# database. Export MLFLOW_TRACKING_URI to override.
TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", _DEFAULT_URI)
MIN_POSITIVE_IN_SAMPLE: int = 20

# MLflow 3.x serialises sklearn objects via skops, which refuses to write any
# type that is not explicitly declared trusted. These are exactly the types this
# project constructs itself (a fitted LightGBM model and a fitted
# TfidfVectorizer) — no third-party or user-supplied class is ever serialised.
MODEL_TRUSTED_TYPES: list[str] = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
]
VECTORIZER_TRUSTED_TYPES: list[str] = ["collections.OrderedDict"]


@task
def load_features_task(
    input_path: Path, limit: int | None = None, seed: int = SPLIT_SEED
) -> pd.DataFrame:
    """Read the feature parquet, optionally as a stratified subsample.

    Reads only the columns training needs (subject, target, 8 features) so
    `uid`, `from`, and `date` are never materialised.

    When `limit` is set the sample is stratified on the target rather than a
    head slice: the parquet is in ingest order, so the first N rows are not
    guaranteed to contain any of the 0.68% positives, which would make a 70/15/15
    stratified split impossible. The sample is topped up to
    `MIN_POSITIVE_IN_SAMPLE` positives so the split has enough of the class.
    """
    logger = get_run_logger()
    column_names = [SUBJECT, TARGET, *FEATURE_COLUMNS]

    logger.info("Reading %s (columns: %s)...", input_path, len(column_names))
    df = pd.read_parquet(input_path, columns=column_names)
    logger.info("Read %d rows", len(df))

    if limit is None or limit >= len(df):
        return df

    frac = limit / len(df)
    parts = [
        group.sample(frac=frac, random_state=seed)
        for _, group in df.groupby(TARGET, sort=True)
    ]
    sampled = pd.concat(parts)
    have = int((sampled[TARGET] == 1).sum())
    if have < MIN_POSITIVE_IN_SAMPLE:
        shortfall = MIN_POSITIVE_IN_SAMPLE - have
        positives = df[df[TARGET] == 1]
        top_up = min(shortfall, len(positives))
        logger.info(
            "Sample had only %d positives; topping up %d to reach %d",
            have,
            top_up,
            MIN_POSITIVE_IN_SAMPLE,
        )
        sampled = pd.concat([sampled, positives.sample(n=top_up, random_state=seed)])

    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    logger.info(
        "Stratified sample: %d rows, %d positives (%.3f%%)",
        len(sampled),
        int((sampled[TARGET] == 1).sum()),
        100.0 * float((sampled[TARGET] == 1).mean()),
    )
    return sampled


@task
def train_variant_task(
    df: pd.DataFrame,
    variant_name: str,
    n_estimators: int = 2000,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_features: int = 100,
    seed: int = SPLIT_SEED,
    stopping_rounds: int = 100,
) -> dict[str, Any]:
    """Train one variant inside its own MLflow run and log everything.

    The split and the TF-IDF fit both happen here, on the train split only
    (ADR-008). The vectorizer is fitted only for variants that use TF-IDF, and
    is logged as its own artifact because the serving layer needs it to
    transform raw subjects at request time.

    Args:
        df: Feature frame from `load_features_task`.
        variant_name: One of the names in `train.VARIANTS`.
        n_estimators: LightGBM boosting rounds (upper bound; early stopping
            usually stops well short).
        learning_rate: Boosting learning rate.
        num_leaves: LightGBM tree complexity.
        max_features: TF-IDF vocabulary size.
        seed: Split and model seed, shared across variants so the split is
            identical for both runs.
        stopping_rounds: Early-stopping patience.

    Returns:
        Summary dict: run_id, variant, split sizes, best_iteration, and the
        validation/test class-1 metrics. Values are cast to builtin types.
    """
    logger = get_run_logger()
    variant = variant_by_name(variant_name)
    logger.info("=== Training variant %s ===", variant.name)
    logger.info("Description: %s", variant.description)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    train_df, val_df, test_df = stratified_split(df, seed=seed)
    logger.info(
        "Split -> train %d / val %d / test %d rows",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    vectorizer = None
    if variant.use_tfidf:
        vectorizer = fit_subject_vectorizer(
            train_df[SUBJECT], max_features=max_features
        )
        logger.info("Fitted subject vectorizer on train split only")

    X_train = build_matrices(
        train_df,
        vectorizer=vectorizer,
        feature_columns=variant.feature_columns,
        use_tfidf=variant.use_tfidf,
    )
    X_val = build_matrices(
        val_df,
        vectorizer=vectorizer,
        feature_columns=variant.feature_columns,
        use_tfidf=variant.use_tfidf,
    )
    X_test = build_matrices(
        test_df,
        vectorizer=vectorizer,
        feature_columns=variant.feature_columns,
        use_tfidf=variant.use_tfidf,
    )
    logger.info("Matrix shape: %d x %d", *X_train.shape)

    model = build_model(
        scale_pos_weight=SCALE_POS_WEIGHT,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=seed,
    )
    model, best_iteration = train_model(
        model,
        X_train,
        train_df[TARGET],
        X_val,
        val_df[TARGET],
        stopping_rounds=stopping_rounds,
    )
    logger.info("Best iteration: %d", best_iteration)

    val_scores = evaluate(model, X_val, val_df[TARGET])
    test_scores = evaluate(model, X_test, test_df[TARGET])
    logger.info(
        "Val  class-1 F1 %.4f P %.4f R %.4f",
        val_scores["f1"],
        val_scores["precision"],
        val_scores["recall"],
    )
    logger.info(
        "Test class-1 F1 %.4f P %.4f R %.4f",
        test_scores["f1"],
        test_scores["precision"],
        test_scores["recall"],
    )

    importance = feature_importance_frame(
        model,
        vectorizer=vectorizer,
        feature_columns=variant.feature_columns,
        use_tfidf=variant.use_tfidf,
    )

    params: dict[str, Any] = {
        "variant": variant.name,
        "seed": seed,
        "scale_pos_weight": SCALE_POS_WEIGHT,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "best_iteration": best_iteration,
        "stopping_rounds": stopping_rounds,
        "use_tfidf": variant.use_tfidf,
        "max_features": max_features if variant.use_tfidf else 0,
        "tabular_features": ",".join(variant.feature_columns),
        "n_matrix_columns": int(X_train.shape[1]),
        "primary_metric": "class_1_f1",
        "lib_lightgbm": _lightgbm_version(),
        "split_seed": seed,
    }
    for split_name, split_df in (
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ):
        rows = len(split_df)
        positives = int((split_df[TARGET] == 1).sum())
        params[f"{split_name}_rows"] = rows
        params[f"{split_name}_positives"] = positives
        params[f"{split_name}_positive_rate"] = round(positives / rows, 6)

    metrics: dict[str, float] = {}
    for split_name, scores in (("val", val_scores), ("test", test_scores)):
        for key, value in scores.items():
            metrics[f"{split_name}_{key}"] = float(value)

    with mlflow.start_run(run_name=variant.name):
        run = mlflow.active_run()
        mlflow.set_tags(
            {
                "variant": variant.name,
                "description": variant.description,
                "known_limitation": LIMITATION_NOTE,
                "labels": "heuristic (ADR-005)",
            }
        )
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model, name="model", skops_trusted_types=MODEL_TRUSTED_TYPES
        )

        if vectorizer is not None:
            mlflow.sklearn.log_model(
                vectorizer,
                name="subject_vectorizer",
                skops_trusted_types=VECTORIZER_TRUSTED_TYPES,
            )

        with tempfile.TemporaryDirectory() as tmp:
            plot_path = plot_feature_importance(
                importance,
                Path(tmp),
                title=variant.name,
            )
            mlflow.log_artifact(str(plot_path), artifact_path="feature_importance")

        logger.info("Logged run %s to experiment %s", run.info.run_id, EXPERIMENT_NAME)

    return {
        "run_id": run.info.run_id,
        "variant": variant.name,
        "best_iteration": best_iteration,
        "n_matrix_columns": int(X_train.shape[1]),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "val_f1": float(val_scores["f1"]),
        "val_precision": float(val_scores["precision"]),
        "val_recall": float(val_scores["recall"]),
        "test_f1": float(test_scores["f1"]),
        "test_precision": float(test_scores["precision"]),
        "test_recall": float(test_scores["recall"]),
        "test_roc_auc": float(test_scores.get("roc_auc", float("nan"))),
    }


def _lightgbm_version() -> str:
    """Return the installed LightGBM version for run reproducibility."""
    try:
        import lightgbm

        return lightgbm.__version__
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def compare_variants(results: list[dict[str, Any]]) -> str:
    """Render a short side-by-side comparison of two training variants.

    Class-1 F1 is the primary metric; the other columns are context, not
    alternatives to it.
    """
    lines = [
        f"{'variant':<24} {'test F1':>9} {'precision':>10} {'recall':>9} {'roc_auc':>8}"
    ]
    for result in results:
        lines.append(
            f"{result['variant']:<24} "
            f"{result['test_f1']:>9.4f} "
            f"{result['test_precision']:>10.4f} "
            f"{result['test_recall']:>9.4f} "
            f"{result['test_roc_auc']:>8.4f}"
        )
    return "\n".join(lines)
