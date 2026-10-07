"""Tests for src.evaluation.evaluate (pure functions — no MLflow, no Evidently)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import precision_recall_curve

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
    threshold_frame,
    verify_split,
)

# Hand-made example with one tied score pair (0.5 twice).
TOY_Y = np.array([0, 0, 1, 1, 0, 1])
TOY_S = np.array([0.1, 0.5, 0.5, 0.9, 0.2, 0.7])

# Heavy ties: only three distinct score values. Mirrors the best_iteration=1
# regime, where a near-stump model emits a handful of distinct outputs.
TIED_Y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
TIED_S = np.array([0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.5, 0.5])


def test_sweep_matches_sklearn_pr_curve() -> None:
    precision, recall, thresholds = precision_recall_curve(TOY_Y, TOY_S)
    frame = sweep_thresholds(TOY_Y, TOY_S)
    assert frame["threshold"].tolist() == pytest.approx(list(thresholds))
    # sklearn appends a final (precision=1, recall=0) point with no threshold.
    assert frame["precision"].tolist() == pytest.approx(list(precision[:-1]))
    assert frame["recall"].tolist() == pytest.approx(list(recall[:-1]))


def test_sweep_thresholds_ascending_and_complete() -> None:
    frame = sweep_thresholds(TOY_Y, TOY_S)
    assert frame["threshold"].tolist() == [0.1, 0.2, 0.5, 0.7, 0.9]
    assert (frame["tp"] + frame["fp"] + frame["tn"] + frame["fn"] == 6).all()


def test_sweep_heavy_ties_exact_counts() -> None:
    frame = sweep_thresholds(TIED_Y, TIED_S)
    assert frame["threshold"].tolist() == [0.2, 0.5, 0.8]

    at_low = frame.iloc[0]
    assert (at_low["tp"], at_low["fp"], at_low["tn"], at_low["fn"]) == (4, 4, 0, 0)
    assert at_low["precision"] == pytest.approx(0.5)
    assert at_low["recall"] == pytest.approx(1.0)
    assert at_low["f1"] == pytest.approx(2 / 3)

    at_mid = frame.iloc[1]
    assert (at_mid["tp"], at_mid["fp"], at_mid["tn"], at_mid["fn"]) == (3, 2, 2, 1)
    assert at_mid["precision"] == pytest.approx(0.6)
    assert at_mid["recall"] == pytest.approx(0.75)
    assert at_mid["f1"] == pytest.approx(2 / 3)
    assert at_mid["predicted_positives"] == 5
    assert at_mid["flag_rate"] == pytest.approx(0.625)

    at_high = frame.iloc[2]
    assert (at_high["tp"], at_high["fp"], at_high["tn"], at_high["fn"]) == (2, 1, 3, 2)
    assert at_high["precision"] == pytest.approx(2 / 3)
    assert at_high["recall"] == pytest.approx(0.5)
    assert at_high["f1"] == pytest.approx(4 / 7)


def test_threshold_frame_keeps_given_order() -> None:
    frame = threshold_frame(TIED_Y, TIED_S, [0.8, 0.2])
    assert frame["threshold"].tolist() == [0.8, 0.2]
    assert frame.iloc[0]["tp"] == 2
    assert frame.iloc[1]["tp"] == 4


def test_metrics_at_threshold_exact_values() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.2, 0.3, 0.7, 0.8])
    result = metrics_at_threshold(y_true, y_proba, threshold=0.5)
    assert result == {
        "threshold": 0.5,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "tp": 2,
        "fp": 0,
        "tn": 2,
        "fn": 0,
        "predicted_positives": 2,
        "flag_rate": 0.5,
    }


def test_metrics_at_threshold_partial_recall() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.2, 0.3, 0.7, 0.8])
    result = metrics_at_threshold(y_true, y_proba, threshold=0.75)
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (1, 0, 2, 1)
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["f1"] == pytest.approx(2 / 3)


def test_zero_division_returns_zero_not_nan() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.2, 0.3, 0.7, 0.8])
    # Threshold above every score: nothing predicted positive.
    frame = threshold_frame(y_true, y_proba, [2.0])
    row = frame.iloc[0]
    assert (row["tp"], row["fp"], row["tn"], row["fn"]) == (0, 0, 2, 2)
    assert row["precision"] == 0.0
    assert row["recall"] == 0.0
    assert row["f1"] == 0.0
    assert not frame.isna().any().any()

    result = metrics_at_threshold(y_true, y_proba, threshold=2.0)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_zero_division_no_positives_in_labels() -> None:
    frame = threshold_frame(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]), [0.15])
    row = frame.iloc[0]
    assert (row["tp"], row["fp"], row["tn"], row["fn"]) == (0, 2, 1, 0)
    assert row["recall"] == 0.0
    assert row["f1"] == 0.0
    assert not frame.isna().any().any()


def test_threshold_chosen_on_val_applied_unchanged_to_test() -> None:
    y_val = np.array([0, 0, 1, 1])
    s_val = np.array([0.1, 0.2, 0.8, 0.9])
    y_test = np.array([0, 1, 1, 1])
    s_test = np.array([0.85, 0.1, 0.2, 0.3])

    val_frame = sweep_thresholds(y_val, s_val)
    best = float(val_frame.loc[val_frame["f1"].idxmax(), "threshold"])
    assert best == 0.8

    # The test split would choose differently on its own (0.1, F1 = 6/7),
    # which is exactly why the choice must come from val.
    test_frame = sweep_thresholds(y_test, s_test)
    test_best = float(test_frame.loc[test_frame["f1"].idxmax(), "threshold"])
    assert test_best == pytest.approx(0.1)
    assert best != test_best

    result = metrics_at_threshold(y_test, s_test, threshold=best)
    assert result["threshold"] == best
    assert result["predicted_positives"] == 1
    assert result["f1"] == 0.0


def test_pr_auc_perfect_ranking_is_one() -> None:
    assert pr_auc(np.array([0, 1, 1]), np.array([0.1, 0.4, 0.9])) == 1.0


def test_pr_auc_partial_ranking() -> None:
    # Hand-derived step integral: (0.5 - 0) * 1 + (1 - 0.5) * 2/3 = 5/6.
    assert pr_auc(
        np.array([0, 1, 0, 1]), np.array([0.1, 0.2, 0.3, 0.4])
    ) == pytest.approx(5 / 6)


def test_pr_auc_raises_with_no_positives() -> None:
    with pytest.raises(ValueError, match="no positive labels"):
        pr_auc(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))


def test_roc_auc_perfect_ranking_is_one() -> None:
    assert roc_auc(np.array([0, 1, 1]), np.array([0.1, 0.4, 0.9])) == 1.0


def test_roc_auc_partial_ranking() -> None:
    # 3 of 4 positive/negative pairs concordant.
    assert roc_auc(np.array([0, 1, 0, 1]), np.array([0.1, 0.2, 0.3, 0.4])) == 0.75


def test_roc_auc_raises_on_single_class() -> None:
    with pytest.raises(ValueError, match="single label class"):
        roc_auc(np.array([1, 1, 1]), np.array([0.6, 0.7, 0.8]))


def test_pr_auc_by_iteration_values_and_order() -> None:
    y_true = np.array([0, 0, 1, 1])
    proba = {
        10: np.array([0.1, 0.9, 0.4, 0.6]),
        1: np.array([0.2, 0.3, 0.7, 0.8]),
    }
    frame = pr_auc_by_iteration(y_true, proba)
    assert list(frame.columns) == ["iteration", "pr_auc", "roc_auc"]
    assert frame["iteration"].tolist() == [1, 10]
    assert frame["pr_auc"].tolist() == [
        pr_auc(y_true, proba[1]),
        pr_auc(y_true, proba[10]),
    ]
    assert frame["roc_auc"].tolist() == [
        roc_auc(y_true, proba[1]),
        roc_auc(y_true, proba[10]),
    ]


def test_pr_auc_by_iteration_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        pr_auc_by_iteration(np.array([0, 1]), {})


def test_pr_auc_by_iteration_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="1.*scores.*2.*labels"):
        pr_auc_by_iteration(np.array([0, 1]), {1: np.array([0.5]), 5: np.array([0.5])})


def test_verify_split_passes_on_match() -> None:
    reconstructed = {
        "val_rows": 77610,
        "val_positives": 529,
        "test_rows": 77610,
        "test_positives": 531,
    }
    # MLflow stringifies params; the gate must accept that form.
    logged = {key: str(value) for key, value in reconstructed.items()}
    assert verify_split(reconstructed, logged) is None


def test_verify_split_lists_all_mismatches() -> None:
    reconstructed = {
        "val_rows": 77610,
        "val_positives": 529,
        "test_rows": 77610,
        "test_positives": 531,
    }
    logged = dict(reconstructed)
    logged["val_rows"] = 77609
    logged["test_positives"] = 533
    with pytest.raises(SplitMismatchError) as excinfo:
        verify_split(reconstructed, logged)
    message = str(excinfo.value)
    assert "val_rows" in message
    assert "test_positives" in message
    assert "val_positives" not in message
    assert "test_rows" not in message


def test_verify_split_raises_on_missing_logged_key() -> None:
    reconstructed = dict.fromkeys(SPLIT_CHECK_KEYS, 1)
    logged = dict(reconstructed)
    del logged["test_rows"]
    with pytest.raises(SplitMismatchError, match="test_rows"):
        verify_split(reconstructed, logged)


def test_verify_split_raises_on_missing_reconstructed_key() -> None:
    logged = dict.fromkeys(SPLIT_CHECK_KEYS, 1)
    with pytest.raises(SplitMismatchError, match="val_rows"):
        verify_split({}, logged)


def test_verify_split_raises_on_non_integer_logged_value() -> None:
    reconstructed = dict.fromkeys(SPLIT_CHECK_KEYS, 1)
    logged = dict(reconstructed)
    logged["val_rows"] = "not-a-number"
    with pytest.raises(SplitMismatchError, match="val_rows"):
        verify_split(reconstructed, logged)


def test_plot_pr_curve_writes_png(tmp_path: Path) -> None:
    output = plot_pr_curve(
        TIED_Y,
        TIED_S,
        tmp_path / "pr.png",
        title="behavioral-only",
        operating_threshold=0.5,
    )
    assert output.suffix == ".png"
    assert output.stat().st_size > 0


def test_plot_pr_curve_without_operating_point(tmp_path: Path) -> None:
    output = plot_pr_curve(TOY_Y, TOY_S, tmp_path / "nested" / "pr.png", title="toy")
    assert output.stat().st_size > 0


def test_plot_roc_curve_writes_png(tmp_path: Path) -> None:
    output = plot_roc_curve(
        TIED_Y,
        TIED_S,
        tmp_path / "roc.png",
        title="behavioral-only",
        operating_threshold=0.5,
    )
    assert output.suffix == ".png"
    assert output.stat().st_size > 0


def test_inputs_must_match_in_length() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        threshold_frame(np.array([0, 1]), np.array([0.5]), [0.5])


def test_inputs_reject_non_finite_scores() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        sweep_thresholds(np.array([0, 1]), np.array([0.5, np.inf]))


def test_inputs_reject_non_binary_labels() -> None:
    with pytest.raises(ValueError, match="binary"):
        sweep_thresholds(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))


def test_thresholds_reject_non_finite() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        threshold_frame(TOY_Y, TOY_S, [0.5, np.nan])
