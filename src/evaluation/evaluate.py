"""Evaluation metrics, threshold sweeps, and split verification.

Pure functions — no Prefect, no MLflow, no Evidently, no model loading. All
orchestration and IO lives in src/tasks/evaluate_tasks.py, following the ADR-008
pattern set by ingest.py, features.py, and train.py.

Two caveats govern every number produced here:

* PR-AUC (average precision) is the PRIMARY metric. At 0.68% positives ROC-AUC
  overstates the model badly (training logged ROC-AUC = 0.85 alongside class-1
  F1 = 0.12), so ROC-AUC is reported only as secondary context.
* Operating thresholds are points on inflated, uncalibrated scores — NOT true
  probabilities. Training uses scale_pos_weight = 145.7 (see SCALE_POS_WEIGHT
  in src/training/train.py), which rescales the positive class and pushes
  predicted scores upward: on validation, behavioral-only flags ~5.4% of emails
  at the default 0.5 cut versus a 0.68% actual positive rate. The optimal
  operating threshold is therefore likely above 0.5, possibly above 0.95, which
  is why sweep_thresholds covers every distinct operating point from
  precision_recall_curve instead of a fixed grid.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

#: Columns of the frame returned by threshold_frame / sweep_thresholds.
THRESHOLD_COLUMNS: tuple[str, ...] = (
    "threshold",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "tn",
    "fn",
    "predicted_positives",
    "flag_rate",
)

#: Integer-valued columns of the threshold frame.
_COUNT_COLUMNS: frozenset[str] = frozenset(
    {"tp", "fp", "tn", "fn", "predicted_positives"}
)

#: Split statistics the reproducibility gate compares. These are the exact keys
#: train_tasks.py logs as MLflow params for every variant.
SPLIT_CHECK_KEYS: tuple[str, ...] = (
    "val_rows",
    "val_positives",
    "test_rows",
    "test_positives",
)


class SplitMismatchError(RuntimeError):
    """Raised when the reconstructed split does not match the logged run."""


def _check_labels(y_true: np.ndarray | pd.Series) -> np.ndarray:
    """Coerce and validate binary labels.

    Raises:
        ValueError: if labels are empty or not binary 0/1.
    """
    labels = np.asarray(y_true).ravel()
    if labels.size == 0:
        raise ValueError("empty labels")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("labels must be binary 0/1")
    return labels.astype(int)


def _as_arrays(
    y_true: np.ndarray | pd.Series, y_proba: np.ndarray | pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce and validate labels and scores.

    Raises:
        ValueError: on length mismatch, empty inputs, non-finite scores,
            or non-binary labels.
    """
    labels = _check_labels(y_true)
    scores = np.asarray(y_proba, dtype=np.float64).ravel()
    if labels.shape[0] != scores.shape[0]:
        raise ValueError(
            f"length mismatch: {labels.shape[0]} labels " f"vs {scores.shape[0]} scores"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain non-finite values")
    return labels, scores


def threshold_frame(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """Class-1 metrics at explicit decision thresholds.

    Predicted positive iff score >= threshold. Scores are the inflated,
    uncalibrated model outputs described in the module docstring, not true
    probabilities. Confusion counts are computed exactly in one pass over the
    score-sorted labels (never re-predicting per threshold); divisions by zero
    yield 0.0, matching the zero_division=0 convention in train.py.

    Args:
        y_true: Binary 0/1 labels.
        y_proba: Positive-class scores, one per label.
        thresholds: Operating points to evaluate, in any order. Rows follow
            the given order.

    Returns:
        Frame with columns threshold, precision, recall, f1, tp, fp, tn, fn,
        predicted_positives, flag_rate.

    Raises:
        ValueError: on invalid inputs or non-finite thresholds.
    """
    labels, scores = _as_arrays(y_true, y_proba)
    grid = np.atleast_1d(np.asarray(thresholds, dtype=np.float64))
    if not np.all(np.isfinite(grid)):
        raise ValueError("thresholds contain non-finite values")

    n = scores.size
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    total_pos = int(sorted_labels.sum())

    # First index with score >= t is the predicted-negative count at t.
    idx = np.searchsorted(sorted_scores, grid, side="left")
    positives_below = np.concatenate(([0], np.cumsum(sorted_labels)))[idx]
    false_negative = positives_below
    true_positive = total_pos - false_negative
    predicted_positives = n - idx
    false_positive = predicted_positives - true_positive
    true_negative = idx - false_negative

    precision = np.divide(
        true_positive,
        predicted_positives,
        out=np.zeros_like(grid),
        where=predicted_positives > 0,
    )
    if total_pos > 0:
        recall = true_positive / total_pos
    else:
        recall = np.zeros_like(grid)
    f1_denom = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        f1_denom,
        out=np.zeros_like(grid),
        where=f1_denom > 0,
    )
    return pd.DataFrame(
        {
            "threshold": grid,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": true_positive,
            "fp": false_positive,
            "tn": true_negative,
            "fn": false_negative,
            "predicted_positives": predicted_positives,
            "flag_rate": predicted_positives / n,
        }
    )[list(THRESHOLD_COLUMNS)]


def sweep_thresholds(
    y_true: np.ndarray | pd.Series, y_proba: np.ndarray | pd.Series
) -> pd.DataFrame:
    """Sweep every distinct operating point from the precision-recall curve.

    No fixed grid: thresholds are the distinct model scores returned by
    sklearn's precision_recall_curve, so every point where the prediction set
    changes is evaluated. This matters because the scores are inflated and
    uncalibrated (scale_pos_weight = 145.7 pushes them upward) — a grid centred
    on 0.5 would miss the optimum, which the 5.4%-flagged vs 0.68%-positive
    gap suggests lies above 0.5, possibly above 0.95.

    Returns:
        The threshold_frame at the PR-curve thresholds, ascending.
    """
    labels, scores = _as_arrays(y_true, y_proba)
    _, _, thresholds = precision_recall_curve(labels, scores)
    return threshold_frame(labels, scores, thresholds)


def metrics_at_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Class-1 metrics at a single operating threshold on the model scores.

    Returns:
        Dict with threshold, precision, recall, f1, tp, fp, tn, fn,
        predicted_positives, flag_rate.
    """
    frame = threshold_frame(y_true, y_proba, [threshold])
    row = frame.iloc[0]
    result: dict[str, float | int] = {}
    for column in THRESHOLD_COLUMNS:
        value = row[column]
        result[column] = int(value) if column in _COUNT_COLUMNS else float(value)
    return result


def pr_auc(y_true: np.ndarray | pd.Series, y_proba: np.ndarray | pd.Series) -> float:
    """Area under the precision-recall curve as average precision (AP).

    AP is the PRIMARY metric for this project: at 0.68% prevalence it reflects
    the ranking quality the operating point will actually see, where ROC-AUC
    overstates the model. Baseline (random classifier) = prevalence.

    Raises:
        ValueError: if there are no positive labels (sklearn returns 0.0 with
            a warning; we fail loudly instead).
    """
    labels, scores = _as_arrays(y_true, y_proba)
    if not np.any(labels == 1):
        raise ValueError("pr_auc is undefined with no positive labels")
    return float(average_precision_score(labels, scores))


def roc_auc(y_true: np.ndarray | pd.Series, y_proba: np.ndarray | pd.Series) -> float:
    """Area under the ROC curve. Secondary-context metric only.

    See the module docstring: at 0.68% prevalence ROC-AUC overstates the model.

    Raises:
        ValueError: if labels contain a single class.
    """
    labels, scores = _as_arrays(y_true, y_proba)
    if np.unique(labels).size < 2:
        raise ValueError("roc_auc is undefined with a single label class")
    return float(roc_auc_score(labels, scores))


def plot_pr_curve(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    output_path: Path,
    *,
    title: str,
    operating_threshold: float | None = None,
) -> Path:
    """Render the precision-recall curve with a prevalence baseline to a PNG.

    Uses the Agg backend, so it works headless in WSL/Docker.

    Returns:
        Path to the written PNG.
    """
    labels, scores = _as_arrays(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(labels, scores)
    prevalence = float(np.mean(labels))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(7, 5.5))
    axes.plot(recall, precision, color="#3b6ea5", lw=1.8, label="PR curve")
    axes.axhline(
        prevalence,
        color="#888888",
        ls="--",
        lw=1.0,
        label=f"prevalence = {prevalence:.4f}",
    )
    if operating_threshold is not None:
        point = metrics_at_threshold(labels, scores, threshold=operating_threshold)
        axes.scatter(
            [point["recall"]],
            [point["precision"]],
            color="#c1442e",
            zorder=5,
            label=f"threshold = {operating_threshold:.4f}",
        )
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.set_xlabel("Recall (positive class)")
    axes.set_ylabel("Precision (positive class)")
    axes.set_title(title)
    axes.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_roc_curve(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    output_path: Path,
    *,
    title: str,
    operating_threshold: float | None = None,
) -> Path:
    """Render the ROC curve with a chance diagonal to a PNG.

    Uses the Agg backend, so it works headless in WSL/Docker.

    Returns:
        Path to the written PNG.
    """
    labels, scores = _as_arrays(y_true, y_proba)
    fpr, tpr, _ = roc_curve(labels, scores)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(6, 6))
    axes.plot(fpr, tpr, color="#3b6ea5", lw=1.8, label="ROC curve")
    axes.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="#888888",
        ls="--",
        lw=1.0,
        label="chance",
    )
    if operating_threshold is not None:
        point = metrics_at_threshold(labels, scores, threshold=operating_threshold)
        denom = point["fp"] + point["tn"]
        fpr_point = point["fp"] / denom if denom > 0 else 0.0
        axes.scatter(
            [fpr_point],
            [point["recall"]],
            color="#c1442e",
            zorder=5,
            label=f"threshold = {operating_threshold:.4f}",
        )
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.set_xlabel("False positive rate")
    axes.set_ylabel("True positive rate (recall)")
    axes.set_title(title)
    axes.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def verify_split(
    reconstructed: Mapping[str, int],
    logged: Mapping[str, Any],
) -> None:
    """Split-reproducibility safety gate: reconstructed split vs training run.

    Compares val_rows / val_positives / test_rows / test_positives of a locally
    reconstructed stratified_split(seed=42) against the same values logged as
    MLflow params by the training run. The caller fetches the logged params
    (MLflow access stays in evaluate_tasks.py); this stays pure and testable.

    Logged values are coerced with int(float(str(...))) because MLflow
    stringifies params. Counts here are far below float's exact-integer range,
    so the coercion is lossless.

    Raises:
        SplitMismatchError: on any missing key, any non-integer logged value,
            or any count mismatch. Lists every mismatch, not just the first.
    """
    mismatches: list[str] = []
    for key in SPLIT_CHECK_KEYS:
        if key not in reconstructed:
            raise SplitMismatchError(f"reconstructed split is missing {key!r}")
        if key not in logged:
            raise SplitMismatchError(
                f"training run params are missing {key!r}; cannot verify the split"
            )
        try:
            logged_value = int(float(str(logged[key])))
        except (TypeError, ValueError) as exc:
            raise SplitMismatchError(
                f"logged {key}={logged[key]!r} is not an integer"
            ) from exc
        rebuilt_value = int(reconstructed[key])
        if rebuilt_value != logged_value:
            mismatches.append(
                f"{key}: reconstructed={rebuilt_value} logged={logged_value}"
            )
    if mismatches:
        detail = "\n  ".join(mismatches)
        raise SplitMismatchError(
            "split reproducibility gate FAILED -- aborting evaluation:\n" f"  {detail}"
        )


def pr_auc_by_iteration(
    y_true: np.ndarray | pd.Series,
    proba_by_iteration: dict[int, np.ndarray],
) -> pd.DataFrame:
    """PR-AUC and ROC-AUC per boosting-iteration snapshot of the same data.

    Answers the carried-forward best_iteration=1 question for behavioral-only:
    if PR-AUC keeps rising past the early-stopped best iteration, the model is
    under-trained and F1 = 0.12 / ROC-AUC = 0.85 are a floor, not final. If
    PR-AUC is flat across iterations, more trees add nothing and the threshold
    sweep is the only remaining lever. A threshold sweep alone cannot answer
    this — it only moves along one model's ranking.

    Args:
        y_true: Binary 0/1 labels, shared by every snapshot.
        proba_by_iteration: Mapping of boosting-iteration count to
            positive-class probabilities at that snapshot.

    Returns:
        Frame with columns iteration, pr_auc, roc_auc, sorted by iteration.

    Raises:
        ValueError: if the mapping is empty or a snapshot has the wrong length
            (plus the single-class guards of pr_auc / roc_auc).
    """
    labels = _check_labels(y_true)
    if not proba_by_iteration:
        raise ValueError("proba_by_iteration is empty")
    rows = []
    for iteration in sorted(proba_by_iteration):
        scores = np.asarray(proba_by_iteration[iteration], dtype=np.float64).ravel()
        if scores.shape[0] != labels.shape[0]:
            raise ValueError(
                f"iteration {iteration}: {scores.shape[0]} scores "
                f"vs {labels.shape[0]} labels"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"iteration {iteration}: non-finite scores")
        rows.append(
            {
                "iteration": int(iteration),
                "pr_auc": pr_auc(labels, scores),
                "roc_auc": roc_auc(labels, scores),
            }
        )
    return pd.DataFrame(rows, columns=["iteration", "pr_auc", "roc_auc"])
