"""Tests of paired cluster uncertainty with a fixed held-out operating point."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score

from src.evaluation import uncertainty
from src.evaluation.uncertainty import (
    _resample_weights,
    cluster_bootstrap_ci,
    leadership_baseline_metrics,
)


def test_leadership_baseline_hand_computed() -> None:
    baseline = leadership_baseline_metrics(
        np.array([1, 0, 1, 0]), np.array([1, 0, 0, 0])
    )
    assert baseline == {
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(2 / 3),
        "pr_auc": 0.75,
    }


def test_whole_groups_have_identical_resampling_multiplicities() -> None:
    rng = MagicMock()
    rng.integers.return_value = np.array([2, 0, 2])
    weights = _resample_weights(np.array([0, 0, 0, 1, 1, 2]), 3, rng)
    # Entire group zero occurs once, group one is absent, group two twice.
    np.testing.assert_array_equal(weights, [1, 1, 1, 0, 0, 2])
    rng.integers.assert_called_once_with(0, 3, size=3)


def test_bootstrap_is_reproducible_and_seed_changes_draws() -> None:
    labels = np.tile([0, 1, 0, 0, 1], 8)
    scores = np.linspace(0.0, 1.0, len(labels))
    leaders = np.tile([0, 0, 1, 0, 0], 8)
    groups = np.repeat(np.arange(20), 2)
    args = (labels, scores, leaders, groups)
    first = cluster_bootstrap_ci(*args, threshold=0.5, seed=42, n_resamples=40)
    assert first == cluster_bootstrap_ci(*args, threshold=0.5, seed=42, n_resamples=40)
    assert first != cluster_bootstrap_ci(*args, threshold=0.5, seed=43, n_resamples=40)


def test_simple_cluster_intervals_bracket_point_estimates() -> None:
    labels = np.tile([0, 1, 1], 10)
    scores = np.tile([0.1, 0.8, 0.9], 10)
    leaders = np.tile([0, 0, 1], 10)
    groups = np.repeat(np.arange(10), 3)
    results = cluster_bootstrap_ci(
        labels,
        scores,
        leaders,
        groups,
        threshold=0.5,
        n_resamples=30,
    )
    for metric in results.values():
        assert metric["lo"] <= metric["point"] <= metric["hi"]
        assert metric["valid"] == 30
        assert metric["skipped"] == 0
    assert results["model_minus_baseline_f1"]["point"] == pytest.approx(1 / 3)
    assert results["model_minus_baseline_pr_auc"]["point"] == pytest.approx(1 / 6)


def test_weighted_bootstrap_agrees_with_expanded_groups_and_sklearn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = MagicMock()
    rng.integers.return_value = np.array([0, 0, 2])
    monkeypatch.setattr(uncertainty.np.random, "default_rng", lambda seed: rng)
    labels = np.array([0, 1, 1, 0, 1, 0])
    scores = np.array([0.2, 0.2, 0.8, 0.8, 0.5, 0.5])
    leaders = np.array([0, 1, 0, 0, 1, 0])
    groups = np.array([0, 0, 1, 1, 2, 2])
    result = cluster_bootstrap_ci(
        labels,
        scores,
        leaders,
        groups,
        threshold=0.5,
        n_resamples=1,
    )
    indices = [0, 1, 0, 1, 4, 5]  # Explicitly expand whole clusters 0, 0, 2.
    sampled_labels, sampled_scores, sampled_leaders = (
        array[indices] for array in (labels, scores, leaders)
    )
    expected_ap = average_precision_score(sampled_labels, sampled_scores)
    expected_delta_ap = expected_ap - average_precision_score(
        sampled_labels, sampled_leaders
    )
    expected_f1 = f1_score(sampled_labels, sampled_scores >= 0.5)
    expected_delta_f1 = expected_f1 - f1_score(sampled_labels, sampled_leaders)
    for name, expected in (
        ("pr_auc", expected_ap),
        ("f1", expected_f1),
        ("model_minus_baseline_pr_auc", expected_delta_ap),
        ("model_minus_baseline_f1", expected_delta_f1),
    ):
        assert result[name]["lo"] == pytest.approx(expected)
        assert result[name]["hi"] == pytest.approx(expected)


def test_skipped_no_positive_resamples_are_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = MagicMock()
    rng.integers.side_effect = [np.array([0, 0]), np.array([1, 1])]
    monkeypatch.setattr(uncertainty.np.random, "default_rng", lambda seed: rng)
    results = cluster_bootstrap_ci(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.1, 0.9, 0.9]),
        np.array([0, 0, 0, 1]),
        np.array([0, 0, 1, 1]),
        threshold=0.5,
        n_resamples=2,
    )
    for metric in results.values():
        assert metric["skipped"] == 1
        assert metric["valid"] == 1


def test_precision_undefined_without_predictions_and_slice_without_positives() -> None:
    results = cluster_bootstrap_ci(
        np.tile([0, 1], 5),
        np.full(10, 0.1),
        np.tile([0, 1], 5),
        np.repeat(np.arange(5), 2),
        threshold=0.5,
        n_resamples=20,
    )
    for name in ("precision", "nolead_pr_auc"):
        assert results[name]["lo"] is None
        assert results[name]["hi"] is None
        assert results[name]["skipped"] == 20
    assert results["f1"]["point"] == 0.0
    assert results["recall"]["point"] == 0.0


def test_bootstrap_does_not_mutate_scores_or_retune_threshold() -> None:
    labels = np.tile([0, 1, 1], 5)
    scores = np.tile([0.1, 0.8, 0.9], 5)
    before = scores.copy()
    threshold = 0.85
    # A threshold change would alter F1; every identical cluster preserves it.
    result = cluster_bootstrap_ci(
        labels,
        scores,
        np.tile([0, 0, 1], 5),
        np.repeat(np.arange(5), 3),
        threshold=threshold,
        n_resamples=20,
    )
    assert threshold == 0.85
    np.testing.assert_array_equal(scores, before)
    assert result["f1"]["point"] == pytest.approx(2 / 3)
    assert result["f1"]["lo"] == pytest.approx(2 / 3)
    assert result["f1"]["hi"] == pytest.approx(2 / 3)


def test_paired_f1_difference_is_zero_for_identical_predictions() -> None:
    labels = np.tile([0, 1, 1], 6)
    leaders = np.tile([0, 0, 1], 6)
    scores = np.tile([0.1, 0.2, 0.9], 6)
    result = cluster_bootstrap_ci(
        labels,
        scores,
        leaders,
        np.repeat(np.arange(6), 3),
        threshold=0.8,
        n_resamples=20,
    )
    assert result["model_minus_baseline_f1"]["lo"] == 0.0
    assert result["model_minus_baseline_f1"]["hi"] == 0.0


def test_bootstrap_all_negative_resamples_are_explicitly_undefined() -> None:
    results = cluster_bootstrap_ci(
        np.zeros(4),
        np.linspace(0, 1, 4),
        np.zeros(4),
        np.array([0, 0, 1, 1]),
        threshold=0.5,
        n_resamples=10,
    )
    for metric in results.values():
        assert metric == {
            "point": None,
            "lo": None,
            "hi": None,
            "valid": 0,
            "skipped": 10,
        }
