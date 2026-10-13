"""Prefect task wrappers and MLflow/Evidently IO around the pure evaluation functions.

This module (with train_tasks.py) is the only layer that touches MLflow or
Evidently; src/evaluation/evaluate.py stays pure (ADR-008).

An evaluation run NEVER writes to the training run. It opens a fresh run named
"evaluation-behavioral-only" in the same experiment, records the source run id
as both a param and a tag, and logs only scalars — the threshold sweep, the
iteration table, the plots, and the Evidently HTML go in as artifacts, never as
individual metrics.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import evidently
import mlflow
import mlflow.sklearn  # noqa: F401  (registers mlflow.sklearn.load_model)
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from lightgbm import LGBMClassifier
from mlflow.tracking import MlflowClient
from prefect import get_run_logger, task

from src.evaluation.evaluate import (
    SPLIT_CHECK_KEYS,
    SplitMismatchError,
    metrics_at_threshold,
    plot_pr_curve,
    plot_roc_curve,
    pr_auc,
    pr_auc_by_iteration,
    roc_auc,
    sweep_thresholds,
    verify_split,
)
from src.tasks.train_tasks import DEDUPE_POLICY, EXPERIMENT_NAME, TRACKING_URI
from src.training.splits import (
    SPLIT_STRATEGY,
    SPLIT_VERSION,
    grouped_split,
    split_fingerprint,
)
from src.training.train import (
    BEHAVIORAL_FEATURES,
    SPLIT_SEED,
    TARGET,
    build_matrices,
    build_model,
    scale_pos_weight_from_labels,
)

#: MLflow run name for every evaluation run (never the training run's name).
EVAL_RUN_NAME: str = "evaluation-behavioral-only"

#: Training run under evaluation (behavioral-only, re-run into SQLite).
DEFAULT_SOURCE_RUN_ID: str = "7189707b198f47748f89965712eee4bc"

#: Name of the LoggedModel holding the fitted classifier in the source run.
MODEL_ARTIFACT_NAME: str = "model"

#: Evidently reference cap: train rows are sampled down to this (approved).
REFERENCE_CAP: int = 50000

#: Boosting-iteration probes for the under-training check, before clamping to
#: the verified [1, total-trees] range. `best` and `total` are added at runtime.
ITERATION_PROBES: tuple[int, ...] = (1, 10, 50)

#: Add the artifact best iteration and final retrained tree at runtime.
DIAGNOSTIC_PROBES: tuple[int, ...] = (1, 10, 50, 100, 300, 600, 1000, 1500)

#: Sanity gate: retrain truncated to artifact trees must reproduce its val PR-AUC.
SANITY_TOLERANCE: float = 1e-6


class DiagnosticMismatchError(RuntimeError):
    """Raised when the diagnostic retrain fails its sanity gate."""


def _source_params(client: MlflowClient, source_run_id: str) -> dict[str, str]:
    """Reject legacy/superseded sources before model scoring or diagnostic fitting."""
    source = client.get_run(source_run_id)
    params = dict(source.data.params)
    missing = [key for key in SPLIT_CHECK_KEYS if key not in params]
    if "superseded" in source.data.tags or missing:
        raise SplitMismatchError(
            f"Source run {source_run_id} is a superseded run from the "
            "duplicate-leaky split (superseded tag or missing split params: "
            f"{missing}). Use clean behavioral-only run "
            f"{DEFAULT_SOURCE_RUN_ID} instead."
        )
    return params


def _split_metadata(splits: dict[str, pd.DataFrame], seed: int) -> dict[str, Any]:
    """Describe existing split frames without reconstructing any assignments."""
    metadata: dict[str, Any] = {
        "split_strategy": SPLIT_STRATEGY,
        "split_version": SPLIT_VERSION,
        "split_seed": seed,
        "dedupe_policy": DEDUPE_POLICY,
        "split_fingerprint": split_fingerprint(
            splits["train"], splits["val"], splits["test"]
        ),
        "scale_pos_weight": scale_pos_weight_from_labels(splits["train"][TARGET]),
    }
    for name in ("train", "val", "test"):
        metadata[f"{name}_rows"] = len(splits[name])
        metadata[f"{name}_positives"] = int((splits[name][TARGET] == 1).sum())
    return metadata


@task
def load_split_task(
    input_path: Path, seed: int = SPLIT_SEED
) -> dict[str, pd.DataFrame]:
    """Read the feature parquet and rebuild the exact training split.

    Read uid/from/subject/priority plus the 7 behavioral columns, never body.
    This is the ONLY evaluation task that invokes grouped_split. The flow
    calls it once and passes its frames to metrics, diagnostic, and Evidently.

    Returns:
        Dict with the `train` / `val` / `test` frames.
    """
    logger = get_run_logger()
    column_names = ["uid", "from", "subject", TARGET, *BEHAVIORAL_FEATURES]
    logger.info("Reading %s (columns: %s)...", input_path, len(column_names))
    df = pd.read_parquet(input_path, columns=column_names)
    logger.info("Read %d rows", len(df))
    train_df, val_df, test_df = grouped_split(df, seed=seed)
    logger.info(
        "Split -> train %d / val %d / test %d rows",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return {"train": train_df, "val": val_df, "test": test_df}


@task
def load_model_task(source_run_id: str = DEFAULT_SOURCE_RUN_ID) -> LGBMClassifier:
    """Load the fitted classifier logged by the source training run.

    MLflow 3 stores sklearn models as LoggedModel entities (under
    `mlruns/<exp>/models/<id>/`), NOT under the run's artifact directory, so
    the model is located by `source_run_id` instead of a run-relative path.

    NOTE on trusted types: `mlflow.sklearn.load_model` in MLflow 3.15 takes no
    `skops_trusted_types` parameter — its signature is `(model_uri, dst_path)`.
    Trust is established at LOG time and recorded in the model's MLmodel file,
    which for this run lists exactly the `MODEL_TRUSTED_TYPES` from
    train_tasks.py (OrderedDict, Booster, LGBMClassifier). Loading therefore
    needs no extra arguments; a load failure here means the logged model is
    untrusted or corrupt, and it fails loudly.

    Returns:
        The fitted LGBMClassifier.
    """
    logger = get_run_logger()
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    _source_params(client, source_run_id)
    experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    candidates = [
        model
        for model in client.search_logged_models(
            experiment_ids=[experiment_id],
            filter_string=f"source_run_id = '{source_run_id}'",
        )
        if model.name == MODEL_ARTIFACT_NAME
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one {MODEL_ARTIFACT_NAME!r} model for run "
            f"{source_run_id}, found {len(candidates)}"
        )
    model_uri = f"models:/{candidates[0].model_id}"
    logger.info("Loading model %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)
    try:
        best_raw = model.best_iteration_
    except Exception:  # LightGBM raises if fit ran without early stopping
        best_raw = None
    total_trees = int(model.booster_.num_trees())
    logger.info(
        "Loaded %s: best_iteration_=%s, booster trees=%d",
        type(model).__name__,
        best_raw,
        total_trees,
    )
    if total_trees == 1:
        logger.warning(
            "Artifact contains a single tree: saved-model iteration analysis "
            "has one row; later trees require the diagnostic retrain."
        )
    return model


@task
def evaluate_variant_task(
    model: LGBMClassifier,
    splits: dict[str, pd.DataFrame],
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    reference_cap: int = REFERENCE_CAP,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """Score the model, tune the threshold on val, report on test, log all.

    Order (frozen): fetch source params and gate the split -> score val and
    test ONCE each -> pick the operating threshold on val (max class-1 F1) ->
    report test at that unchanged threshold (plus the leadership-free slice,
    ADR-010) -> iteration table. Test is never tuned on. The full val sweep
    goes to the run as a CSV artifact, never as individual metrics; only
    scalars are logged as params/metrics.

    The iteration probes {1, 10, 50, best, total} use verified values only:
    `best` is the loaded model's `best_iteration_` (falling back to the tree
    count when early stopping was not used) and `total` is the booster's
    physical tree count — never best + patience. Candidates outside
    [1, total] are dropped, so a single-tree artifact yields a one-row table
    instead of misleading multi-row output. `predict_proba` scores with the
    first-k trees per `num_iteration` (verified on LightGBM 4.7.0).

    Returns:
        Summary dict with builtin types only: run id, source ids, tuned
        threshold, val/test metrics, leadership-slice metrics and counts,
        prevalence/flag rate, and the iteration table as a record list
        (consumed by the HTML report step).
    """
    logger = get_run_logger()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    source_params = _source_params(client, source_run_id)
    if source_params.get("variant") != "behavioral-only":
        raise ValueError(
            "refusing to evaluate: source run "
            f"{source_run_id} is variant {source_params.get('variant')!r}, "
            "this task only evaluates behavioral-only"
        )

    val_df = splits["val"]
    test_df = splits["test"]
    metadata = _split_metadata(splits, seed)
    verify_split(metadata, source_params)
    logger.info("Split gate passed: reconstructed split matches run %s", source_run_id)

    val_matrix = build_matrices(
        val_df, feature_columns=BEHAVIORAL_FEATURES, use_tfidf=False
    )
    test_matrix = build_matrices(
        test_df, feature_columns=BEHAVIORAL_FEATURES, use_tfidf=False
    )
    y_val = val_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()
    val_proba = np.asarray(model.predict_proba(val_matrix)[:, 1], dtype=float)
    test_proba = np.asarray(model.predict_proba(test_matrix)[:, 1], dtype=float)
    logger.info("Scored val (%d) and test (%d) once each", len(y_val), len(y_test))

    sweep = sweep_thresholds(y_val, val_proba)
    # First maximum wins on F1 ties (thresholds ascend), documented.
    threshold = float(sweep.loc[sweep["f1"].idxmax(), "threshold"])
    logger.info("Operating threshold from val (max class-1 F1): %.4f", threshold)

    val_scores = metrics_at_threshold(y_val, val_proba, threshold=threshold)
    test_scores = metrics_at_threshold(y_test, test_proba, threshold=threshold)
    val_pr_auc = pr_auc(y_val, val_proba)
    test_pr_auc = pr_auc(y_test, test_proba)
    val_roc_auc = roc_auc(y_val, val_proba)
    test_roc_auc = roc_auc(y_test, test_proba)
    test_prevalence = float((y_test == 1).mean())

    # Leadership slice (ADR-010): sender_is_leadership == 1 implies the label,
    # so report the operating point on leadership-free rows too, plus how many
    # leadership rows test holds and how many the model flagged.
    leadership_mask = test_df["sender_is_leadership"].to_numpy() == 1
    test_leadership_rows = int(leadership_mask.sum())
    test_leadership_flagged = int(((test_proba >= threshold) & leadership_mask).sum())
    nolead_scores = metrics_at_threshold(
        y_test[~leadership_mask],
        test_proba[~leadership_mask],
        threshold=threshold,
    )
    nolead_pr_auc = pr_auc(y_test[~leadership_mask], test_proba[~leadership_mask])
    logger.info(
        "Leadership slice: %d leadership rows in test, %d flagged; "
        "non-leadership test F1 %.4f PR-AUC %.4f",
        test_leadership_rows,
        test_leadership_flagged,
        float(nolead_scores["f1"]),
        float(nolead_pr_auc),
    )

    total_trees = int(model.booster_.num_trees())
    if total_trees < 1:
        raise ValueError("loaded model contains no trees")
    try:
        best_raw: Any = model.best_iteration_
    except Exception:  # LightGBM raises if fit ran without early stopping
        best_raw = None
    best_iteration = int(best_raw) if best_raw else total_trees
    iterations = sorted(
        {*ITERATION_PROBES, best_iteration, total_trees}
        & set(range(1, total_trees + 1))
    )
    iteration_proba = {
        count: np.asarray(
            model.predict_proba(val_matrix, num_iteration=count)[:, 1], dtype=float
        )
        for count in iterations
    }
    iteration_table = pr_auc_by_iteration(y_val, iteration_proba)
    logger.info(
        "Iteration PR-AUC on val: %s",
        ", ".join(
            f"{count}={value:.4f}"
            for count, value in zip(
                iteration_table["iteration"], iteration_table["pr_auc"]
            )
        ),
    )

    params: dict[str, Any] = {
        **metadata,
        "source_run_id": source_run_id,
        "variant": "behavioral-only",
        "source_best_iteration": best_iteration,
        "loaded_num_trees": total_trees,
        "tuned_threshold": threshold,
        "evidently_reference_cap": reference_cap,
    }
    metrics: dict[str, float] = {
        "val_precision": float(val_scores["precision"]),
        "val_recall": float(val_scores["recall"]),
        "val_f1": float(val_scores["f1"]),
        "test_precision": float(test_scores["precision"]),
        "test_recall": float(test_scores["recall"]),
        "test_f1": float(test_scores["f1"]),
        "val_pr_auc": float(val_pr_auc),
        "test_pr_auc": float(test_pr_auc),
        "val_roc_auc": float(val_roc_auc),
        "test_roc_auc": float(test_roc_auc),
        "test_flag_rate": float(test_scores["flag_rate"]),
        "test_prevalence": float(test_prevalence),
        "test_nolead_precision": float(nolead_scores["precision"]),
        "test_nolead_recall": float(nolead_scores["recall"]),
        "test_nolead_f1": float(nolead_scores["f1"]),
        "test_nolead_pr_auc": float(nolead_pr_auc),
        "test_leadership_rows": float(test_leadership_rows),
        "test_leadership_flagged": float(test_leadership_flagged),
    }
    for count, pr_value, roc_value in zip(
        iteration_table["iteration"],
        iteration_table["pr_auc"],
        iteration_table["roc_auc"],
    ):
        metrics[f"iter_{int(count)}_pr_auc"] = float(pr_value)
        metrics[f"iter_{int(count)}_roc_auc"] = float(roc_value)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        pr_path = plot_pr_curve(
            y_test,
            test_proba,
            tmpdir / "pr_curve_test.png",
            title="behavioral-only: precision-recall (test)",
            operating_threshold=threshold,
        )
        roc_path = plot_roc_curve(
            y_test,
            test_proba,
            tmpdir / "roc_curve_test.png",
            title="behavioral-only: ROC (test)",
            operating_threshold=threshold,
        )
        sweep_path = tmpdir / "threshold_sweep_val.csv"
        sweep.to_csv(sweep_path, index=False)
        iteration_path = tmpdir / "iteration_pr_auc_val.csv"
        iteration_table.to_csv(iteration_path, index=False)
        with mlflow.start_run(run_name=EVAL_RUN_NAME) as run:
            run_id = run.info.run_id
            mlflow.set_tags(
                {"source_run_id": source_run_id, "variant": "behavioral-only"}
            )
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(pr_path), artifact_path="plots")
            mlflow.log_artifact(str(roc_path), artifact_path="plots")
            mlflow.log_artifact(str(sweep_path), artifact_path="thresholds")
            mlflow.log_artifact(str(iteration_path), artifact_path="iterations")
        logger.info("Logged evaluation run %s", run_id)

    return {
        "run_id": run_id,
        "source_run_id": source_run_id,
        "tuned_threshold": threshold,
        "val": {
            "precision": float(val_scores["precision"]),
            "recall": float(val_scores["recall"]),
            "f1": float(val_scores["f1"]),
            "pr_auc": float(val_pr_auc),
            "roc_auc": float(val_roc_auc),
        },
        "test": {
            "precision": float(test_scores["precision"]),
            "recall": float(test_scores["recall"]),
            "f1": float(test_scores["f1"]),
            "pr_auc": float(test_pr_auc),
            "roc_auc": float(test_roc_auc),
        },
        "test_flag_rate": float(test_scores["flag_rate"]),
        "test_prevalence": float(test_prevalence),
        "leadership_slice": {
            "precision": float(nolead_scores["precision"]),
            "recall": float(nolead_scores["recall"]),
            "f1": float(nolead_scores["f1"]),
            "pr_auc": float(nolead_pr_auc),
        },
        "test_leadership_rows": test_leadership_rows,
        "test_leadership_flagged": test_leadership_flagged,
        "source_best_iteration": best_iteration,
        "loaded_num_trees": total_trees,
        "iterations": [
            {
                "iteration": int(count),
                "pr_auc": float(pr_value),
                "roc_auc": float(roc_value),
            }
            for count, pr_value, roc_value in zip(
                iteration_table["iteration"],
                iteration_table["pr_auc"],
                iteration_table["roc_auc"],
            )
        ],
    }


def summarize_drift(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the dataset/column drift summary from an Evidently snapshot dict.

    Written against the Evidently 0.7.x `Snapshot.dict()` shape
    (`DriftedColumnsCount` plus one `ValueDrift` metric per column).
    Headline count and share come from Evidently, not reconstructed flags.
    Explicit boolean flags take precedence when emitted; otherwise use the
    installed 0.7.21 stattest implementations: distances >= threshold,
    chi-square < threshold, K-S <= threshold. Unknown methods are collected
    and reported together. The per-column table must agree with the headline.

    Raises:
        ValueError: on any unrecognized shape or drift method — a mislabelled
            drift summary is worse than none.
    """
    metrics = snapshot.get("metrics")
    if not isinstance(metrics, list):
        raise ValueError("unrecognized Evidently snapshot: 'metrics' is not a list")

    dataset = [
        metric
        for metric in metrics
        if isinstance(metric, dict)
        and str((metric.get("config") or {}).get("type", "")).endswith(
            "DriftedColumnsCount"
        )
    ]
    if len(dataset) != 1:
        raise ValueError(
            "unrecognized Evidently snapshot: expected one DriftedColumnsCount "
            f"metric, found {len(dataset)}"
        )
    value = dataset[0].get("value") or {}
    count = int(value["count"])
    share = float(value["share"])
    rule = float((dataset[0].get("config") or {}).get("drift_share", 0.5))

    columns: list[dict[str, Any]] = []
    unknown: list[str] = []
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        config = metric.get("config") or {}
        if not str(config.get("type", "")).endswith("ValueDrift"):
            continue
        column = str(config["column"])
        threshold = float(config["threshold"])
        method = str(config.get("method", ""))
        raw_value = metric.get("value")
        flag_sources = [metric]
        if isinstance(raw_value, dict):
            flag_sources.append(raw_value)
            raw_value = raw_value.get("value")
        try:
            stat = float(raw_value)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Evidently ValueDrift for {column!r} has a non-scalar value"
            ) from exc
        explicit_flags = [
            source[key]
            for source in flag_sources
            for key in ("drifted", "drift_detected")
            if key in source
        ]
        if explicit_flags:
            if not all(isinstance(flag, (bool, np.bool_)) for flag in explicit_flags):
                raise ValueError(f"non-boolean explicit drift flag for {column!r}")
            if len({bool(flag) for flag in explicit_flags}) != 1:
                raise ValueError(f"conflicting explicit drift flags for {column!r}")
            is_drifted = bool(explicit_flags[0])
        elif method in (
            "Wasserstein distance (normed)",
            "wasserstein",
            "Jensen-Shannon distance",
            "jensenshannon",
        ):
            is_drifted = stat >= threshold
        elif method in ("chi-square p_value", "chisquare"):
            is_drifted = stat < threshold
        elif method in ("K-S p_value", "ks"):
            is_drifted = stat <= threshold
        else:
            unknown.append(f"{column!r}: {method!r}")
            continue
        columns.append(
            {
                "column": column,
                "method": method,
                "value": stat,
                "threshold": threshold,
                "drifted": bool(is_drifted),
            }
        )
    if unknown:
        raise ValueError(
            "unrecognized Evidently drift methods; cannot label drift: "
            + "; ".join(unknown)
        )
    columns.sort(key=lambda row: row["column"])
    drifted = [row["column"] for row in columns if row["drifted"]]
    if len(drifted) != count:
        raise ValueError(
            f"Evidently count ({count}) disagrees with per-column flags "
            f"({len(drifted)}). Per-column table:\n"
            + pd.DataFrame(columns).to_string(index=False)
        )
    return {
        "n_drifted_columns": count,
        "drift_share": share,
        "dataset_drift": bool(share >= rule),
        "drifted_columns": drifted,
        "total_columns": len(columns),
        "columns": columns,
    }


@task
def evidently_baseline_task(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    eval_run_id: str,
    reference_cap: int = REFERENCE_CAP,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """Evidently train-vs-test drift baseline, logged to the evaluation run.

    Reference is the train split capped at `reference_cap` rows (seeded sample;
    the cap is deliberate — a full 362k-row reference OOM'd a 5.6GB host during
    EDA — and is disclosed, not hidden). Current is the full test split. Only
    the 7 behavioral features go in, matching the evaluated model.

    This task runs AFTER all metrics and plots are logged (the flow orders it
    last), so an Evidently failure cannot lose the evaluation results. It
    resumes the evaluation run — never the training run — to attach the HTML
    report and drift scalars.

    Returns:
        Drift summary dict (counts, share, dataset flag, drifted columns,
        reference/current row counts).
    """
    logger = get_run_logger()
    mlflow.set_tracking_uri(TRACKING_URI)
    reference = train_df[list(BEHAVIORAL_FEATURES)]
    if len(reference) > reference_cap:
        logger.info(
            "Capping Evidently reference %d -> %d rows (seed %d)",
            len(reference),
            reference_cap,
            seed,
        )
        reference = reference.sample(n=reference_cap, random_state=seed).reset_index(
            drop=True
        )
    current = test_df[list(BEHAVIORAL_FEATURES)]
    logger.info(
        "Evidently reference=%d rows, current=%d rows", len(reference), len(current)
    )

    snapshot = Report(metrics=[DataDriftPreset()]).run(current, reference)
    summary = summarize_drift(snapshot.dict())
    summary["reference_rows"] = len(reference)
    summary["current_rows"] = len(current)
    logger.info(
        "Drift: %d/%d columns (share %.3f, dataset_drift=%s): %s",
        summary["n_drifted_columns"],
        summary["total_columns"],
        summary["drift_share"],
        summary["dataset_drift"],
        summary["drifted_columns"],
    )

    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "evidently_drift.html"
        snapshot.save_html(str(html_path))
        columns_path = Path(tmp) / "evidently_columns.csv"
        pd.DataFrame(summary["columns"]).to_csv(columns_path, index=False)
        with mlflow.start_run(run_id=eval_run_id):
            mlflow.log_artifact(str(html_path), artifact_path="evidently")
            mlflow.log_artifact(str(columns_path), artifact_path="evidently")
            mlflow.log_metrics(
                {
                    "evidently_n_drifted_columns": float(summary["n_drifted_columns"]),
                    "evidently_drift_share": float(summary["drift_share"]),
                    "evidently_dataset_drift": float(summary["dataset_drift"]),
                }
            )
            mlflow.log_params(
                {
                    "evidently_reference_rows": len(reference),
                    "evidently_reference_seed": seed,
                    "evidently_version": evidently.__version__,
                }
            )
    return summary


@task
def retrain_diagnostic_task(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_run_id: str,
    artifact_val_pr_auc: float,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    *,
    artifact_num_trees: int,
    artifact_best_iteration: int,
) -> dict[str, Any]:
    """Replay source params/round limit without early stopping; score VAL ONLY.

    Reuse the verified flow's train/val frames, never reconstruct a split.
    Retrain with logged seed, learning_rate, num_leaves, scale_pos_weight and
    n_estimators, no eval set or callbacks. The in-memory booster is never saved.
    Gate at the artifact's PHYSICAL tree count: truncated retrain val PR-AUC
    must reproduce the artifact's val PR-AUC within 1e-6 before ANY logging.
    On mismatch retry once with deterministic=True and num_threads=1; report
    both attempts and abort if the unchanged tolerance is still exceeded.

    Metric probes: 1/10/50/100/300/600/1000/1500, artifact best iteration and
    final retrained tree, filtered to existing trees. CSV also includes every
    100th round and the reproduction point. Leadership remains a model input;
    slice scores exclude only leadership==1 rows. Undefined slice scores are
    explicitly marked and omitted from numeric MLflow metrics.
    """
    logger = get_run_logger()
    if not np.isfinite(artifact_val_pr_auc):
        raise DiagnosticMismatchError(
            f"artifact val PR-AUC is not finite ({artifact_val_pr_auc!r}); "
            "cannot gate the diagnostic"
        )
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    source_params = _source_params(client, source_run_id)
    required = (
        "seed",
        "learning_rate",
        "num_leaves",
        "scale_pos_weight",
        "n_estimators",
    )
    missing = [key for key in required if key not in source_params]
    if missing:
        raise ValueError(
            f"source run {source_run_id} is missing training params: {missing}"
        )
    try:
        seed = int(source_params["seed"])
        learning_rate = float(source_params["learning_rate"])
        num_leaves = int(source_params["num_leaves"])
        scale_pos_weight = float(source_params["scale_pos_weight"])
        n_estimators = int(source_params["n_estimators"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"source run {source_run_id} has non-numeric training params"
        ) from exc
    if not 1 <= artifact_num_trees <= n_estimators:
        raise ValueError("artifact tree count must be within source n_estimators")
    if not 1 <= artifact_best_iteration <= artifact_num_trees:
        raise ValueError("artifact best iteration must be within its tree count")

    train_matrix = build_matrices(
        train_df, feature_columns=BEHAVIORAL_FEATURES, use_tfidf=False
    )
    val_matrix = build_matrices(
        val_df, feature_columns=BEHAVIORAL_FEATURES, use_tfidf=False
    )
    y_val = val_df[TARGET].to_numpy()
    nolead_mask = val_df["sender_is_leadership"].to_numpy() == 0
    if len(y_val) == 0 or not np.any(y_val == 1):
        raise DiagnosticMismatchError(
            "all-row validation PR-AUC is undefined; cannot evaluate artifact gate"
        )

    def fit_and_check(deterministic: bool) -> tuple[Any, int, np.ndarray, float]:
        model = build_model(
            scale_pos_weight=scale_pos_weight,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=seed,
        )
        if deterministic:
            model.set_params(deterministic=True, num_threads=1, n_jobs=1)
        logger.info(
            "Diagnostic retrain: %d rounds, no eval set/early stopping; "
            "deterministic=%s",
            n_estimators,
            deterministic,
        )
        model.fit(train_matrix, train_df[TARGET])
        total = int(model.booster_.num_trees())
        if total < artifact_num_trees:
            raise DiagnosticMismatchError(
                f"retrain has {total} trees; cannot reproduce artifact's "
                f"{artifact_num_trees}"
            )
        proba = np.asarray(
            model.predict_proba(val_matrix, num_iteration=artifact_num_trees)[:, 1],
            dtype=float,
        )
        score = pr_auc(y_val, proba)
        gap = abs(score - float(artifact_val_pr_auc))
        logger.info(
            "Artifact-tree gate at %d: retrain %.6f, artifact %.6f, "
            "gap %.2e (limit %.0e)",
            artifact_num_trees,
            score,
            artifact_val_pr_auc,
            gap,
            SANITY_TOLERANCE,
        )
        return model, total, proba, gap

    model, actual_trees, reproduced_proba, gap = fit_and_check(False)
    initial_gap = gap
    deterministic_retry = gap > SANITY_TOLERANCE
    if deterministic_retry:
        logger.warning(
            "Artifact reproduction failed (gap %.2e); "
            "retrying deterministic=True, num_threads=1",
            gap,
        )
        model, actual_trees, reproduced_proba, gap = fit_and_check(True)
    if gap > SANITY_TOLERANCE:
        raise DiagnosticMismatchError(
            "diagnostic retrain diverged from the saved artifact at "
            f"{artifact_num_trees} trees: initial gap {initial_gap:.2e}, "
            f"deterministic single-thread retry gap {gap:.2e} > "
            f"tolerance {SANITY_TOLERANCE:.0e}. Aborting before logging; "
            "do not loosen the tolerance without approval."
        )

    metric_iterations = sorted(
        k
        for k in {*DIAGNOSTIC_PROBES, artifact_best_iteration, actual_trees}
        if 1 <= k <= actual_trees
    )
    snapshots = sorted(
        set(metric_iterations)
        | set(range(100, actual_trees + 1, 100))
        | {artifact_num_trees}
    )
    nolead_labels = y_val[nolead_mask]
    nolead_prevalence = (
        float((nolead_labels == 1).mean()) if len(nolead_labels) else None
    )
    nolead_status = (
        "empty_slice"
        if len(nolead_labels) == 0
        else "no_positive_labels" if not np.any(nolead_labels == 1) else "defined"
    )
    rows = []
    for count in snapshots:
        scores = (
            reproduced_proba
            if count == artifact_num_trees
            else np.asarray(
                model.predict_proba(val_matrix, num_iteration=count)[:, 1], dtype=float
            )
        )
        rows.append(
            {
                "iteration": count,
                "pr_auc": pr_auc(y_val, scores),
                "nolead_pr_auc": (
                    pr_auc(nolead_labels, scores[nolead_mask])
                    if nolead_status == "defined"
                    else None
                ),
            }
        )
    table = pd.DataFrame(rows)
    slice_summaries: dict[str, Any] = {}
    for name, score_column, labels, status in (
        ("all", "pr_auc", y_val, "defined"),
        ("nolead", "nolead_pr_auc", nolead_labels, nolead_status),
    ):
        defined = status == "defined"
        best_row = table.loc[table[score_column].idxmax()] if defined else None
        slice_summaries[name] = {
            "status": status,
            "rows": len(labels),
            "positives": int((labels == 1).sum()),
            "best_iteration": int(best_row["iteration"]) if defined else None,
            "best_pr_auc": float(best_row[score_column]) if defined else None,
            "first_pr_auc": float(table[score_column].iloc[0]) if defined else None,
            "final_pr_auc": float(table[score_column].iloc[-1]) if defined else None,
        }

    metrics = {
        "diagnostic_sanity_gap": float(gap),
        "diagnostic_initial_sanity_gap": float(initial_gap),
    }
    if nolead_prevalence is not None:
        metrics["diagnostic_nolead_prevalence"] = nolead_prevalence
    for row in rows:
        count = row["iteration"]
        if count not in metric_iterations:
            continue
        metrics[f"diagnostic_retrain_{count}_pr_auc"] = float(row["pr_auc"])
        if row["nolead_pr_auc"] is not None:
            metrics[f"diagnostic_retrain_{count}_nolead_pr_auc"] = float(
                row["nolead_pr_auc"]
            )

    with tempfile.TemporaryDirectory() as tmp:
        table_path = Path(tmp) / "retrain_pr_auc_val.csv"
        table.assign(
            nolead_prevalence=nolead_prevalence,
            nolead_status=nolead_status,
        ).to_csv(table_path, index=False)
        with mlflow.start_run(run_id=eval_run_id):
            mlflow.set_tags({"diagnostic_retrain": "true"})
            mlflow.log_params(
                {
                    "diagnostic_n_estimators": n_estimators,
                    "diagnostic_seed": seed,
                    "diagnostic_learning_rate": learning_rate,
                    "diagnostic_num_leaves": num_leaves,
                    "diagnostic_scale_pos_weight": scale_pos_weight,
                    "diagnostic_all_status": "defined",
                    "diagnostic_nolead_status": nolead_status,
                    "diagnostic_sanity_iteration": artifact_num_trees,
                    "diagnostic_actual_trees": actual_trees,
                    "diagnostic_deterministic_retry": deterministic_retry,
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(table_path), artifact_path="diagnostics")
        logger.info("Logged diagnostic retrain on evaluation run %s", eval_run_id)

    return {
        "eval_run_id": eval_run_id,
        "n_estimators": n_estimators,
        "actual_trees": actual_trees,
        "snapshots": snapshots,
        "metric_iterations": metric_iterations,
        "nolead_prevalence": nolead_prevalence,
        "sanity_gap": float(gap),
        "initial_sanity_gap": float(initial_gap),
        "sanity_iteration": artifact_num_trees,
        "deterministic_retry": deterministic_retry,
        "slices": slice_summaries,
        "hyperparams": {
            "seed": seed,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "scale_pos_weight": scale_pos_weight,
        },
        "iterations": rows,
    }
