"""Leadership-rule baseline and fixed-threshold test cluster bootstrap.

Groups are sampled with replacement, never individual rows. Integer row
weights represent each group's multiplicity, exactly equivalent to expanding
the sampled groups. Scores are sorted once for weighted average precision;
memory is O(rows + groups + B), not a B-by-rows resampling matrix.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation.evaluate import metrics_at_threshold, pr_auc

_METRICS = (
    "f1",
    "precision",
    "recall",
    "pr_auc",
    "nolead_pr_auc",
    "model_minus_baseline_f1",
    "model_minus_baseline_pr_auc",
)


def leadership_baseline_metrics(
    y_true: np.ndarray, leadership: np.ndarray
) -> dict[str, float]:
    """Test baseline with binary leadership scores and a fixed 0.5 decision cut.

    PR-AUC uses the 0/1 scores, including ties, not a fitted classifier.
    Raises ValueError for non-binary scores or undefined PR-AUC (no positives).
    """
    scores = np.asarray(leadership).ravel()
    if not np.all(np.isin(scores, (0, 1))):
        raise ValueError("leadership scores must be binary 0/1")
    metrics = metrics_at_threshold(y_true, scores, threshold=0.5)
    return {name: float(metrics[name]) for name in ("precision", "recall", "f1")} | {
        "pr_auc": pr_auc(y_true, scores)
    }


def _prepare_ap(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, ...]:
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ends = np.flatnonzero(np.r_[ordered_scores[:-1] != ordered_scores[1:], True])
    return order, labels[order], ends


def _weighted_ap(prepared: tuple[np.ndarray, ...], weights: np.ndarray) -> float | None:
    if weights.size == 0:
        return None
    order, positives, ends = prepared
    ordered_weights = weights[order]
    tp = np.cumsum(ordered_weights * positives)[ends]
    if tp[-1] == 0:
        return None
    total = np.cumsum(ordered_weights)[ends]
    precision = np.divide(
        tp, total, out=np.zeros_like(tp, dtype=float), where=total > 0
    )
    # Elementwise reduction avoids threaded BLAS overhead for repeated 1-D AP sums.
    return float((np.diff(np.r_[0, tp]) * precision).sum() / tp[-1])


def _resample_weights(
    inverse: np.ndarray, n_groups: int, rng: np.random.Generator
) -> np.ndarray:
    draws = rng.integers(0, n_groups, size=n_groups)
    return np.bincount(draws, minlength=n_groups)[inverse]


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    model_scores: np.ndarray,
    leadership: np.ndarray,
    group_ids: np.ndarray,
    *,
    threshold: float,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """95% percentile intervals from paired whole-group test resampling.

    Each draw samples G of the G observed groups with replacement; every row
    in a selected group receives the same multiplicity. Model/baseline and
    non-leadership slice share that draw. The supplied val-chosen threshold
    never changes, no model is fitted/scored, and no threshold search occurs.

    Resamples with no positives leave all metrics undefined; precision is
    additionally undefined with no flagged rows. Non-leadership PR-AUC needs
    at least one positive in that slice. Paired differences require both
    component metrics to be defined. Skip counts are per metric, not a shared
    denominator. All-undefined metrics have None bounds and valid=0.

    Returns:
        Metric-keyed point/lo/hi/valid/skipped records. Only percentile bounds
        are returned; they need not bracket the point estimate in general.
    """
    labels = np.asarray(y_true).ravel()
    scores = np.asarray(model_scores, dtype=float).ravel()
    leaders = np.asarray(leadership).ravel()
    groups = np.asarray(group_ids).ravel()
    if not (len(labels) == len(scores) == len(leaders) == len(groups)) or not len(
        labels
    ):
        raise ValueError("bootstrap requires nonempty equal-length inputs")
    if not np.all(np.isin(labels, (0, 1))) or not np.all(np.isin(leaders, (0, 1))):
        raise ValueError("bootstrap labels and leadership must be binary 0/1")
    labels = labels.astype(np.int64)
    leaders = leaders.astype(np.int64)
    if not np.all(np.isfinite(scores)) or not np.isfinite(threshold):
        raise ValueError("bootstrap scores and threshold must be finite")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    unique_groups, inverse = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)
    nolead = leaders == 0
    predictions = scores >= threshold
    model_tp = (labels == 1) & predictions
    base_tp = (labels == 1) & (leaders == 1)
    model_ap = _prepare_ap(labels, scores)
    nolead_ap = _prepare_ap(labels[nolead], scores[nolead]) if nolead.any() else None

    def values(weights: np.ndarray) -> dict[str, float | None]:
        result: dict[str, float | None] = dict.fromkeys(_METRICS)
        positive_count = int(np.dot(weights, labels))
        if positive_count == 0:
            return result
        flagged = int(weights[predictions].sum())
        tp = int(weights[model_tp].sum())
        base_flagged = int(np.dot(weights, leaders))
        base_true_positive = int(weights[base_tp].sum())
        result["f1"] = 2 * tp / (positive_count + flagged)
        result["recall"] = tp / positive_count
        result["precision"] = tp / flagged if flagged else None
        result["pr_auc"] = _weighted_ap(model_ap, weights)
        result["nolead_pr_auc"] = (
            _weighted_ap(nolead_ap, weights[nolead]) if nolead_ap is not None else None
        )
        base_f1 = 2 * base_true_positive / (positive_count + base_flagged)
        # Binary-score AP: top (leadership) tie then the score-zero tie.
        base_precision = base_true_positive / base_flagged if base_flagged else 0.0
        base_recall = base_true_positive / positive_count
        base_ap = base_recall * base_precision + (
            1 - base_recall
        ) * positive_count / int(weights.sum())
        result["model_minus_baseline_f1"] = result["f1"] - base_f1
        result["model_minus_baseline_pr_auc"] = result["pr_auc"] - base_ap
        return result

    point = values(np.ones(len(labels), dtype=int))
    draws: dict[str, list[float]] = {name: [] for name in _METRICS}
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        metrics = values(_resample_weights(inverse, n_groups, rng))
        for name, value in metrics.items():
            if value is not None:
                draws[name].append(value)

    intervals = {}
    for name, samples in draws.items():
        lo, hi = np.percentile(samples, [2.5, 97.5]) if samples else (None, None)
        intervals[name] = {
            "point": point[name],
            "lo": float(lo) if lo is not None else None,
            "hi": float(hi) if hi is not None else None,
            "valid": len(samples),
            "skipped": n_resamples - len(samples),
        }
    return intervals
