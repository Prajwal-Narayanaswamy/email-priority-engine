"""Tests for src.training.train (pure functions — no MLflow, no live server)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, issparse

from src.features.features import FEATURE_COLUMNS, fit_subject_vectorizer
from src.training.train import (
    BEHAVIORAL_FEATURES,
    SPLIT_SEED,
    VARIANTS,
    build_matrices,
    build_model,
    early_stopping_f1,
    evaluate,
    feature_importance_frame,
    plot_feature_importance,
    scale_pos_weight_from_labels,
    stratified_split,
    train_model,
    variant_by_name,
)

POSITIVE_RATE = 0.12
N_ROWS = 240


@pytest.fixture
def feature_frame() -> pd.DataFrame:
    """Synthetic imbalanced feature frame with a learnable subject signal."""
    rng = np.random.default_rng(0)
    n_pos = int(N_ROWS * POSITIVE_RATE)
    n_neg = N_ROWS - n_pos

    subjects = [
        "urgent deadline action required please respond asap" for _ in range(n_pos)
    ] + [f"weekly status update number {i} for the team" for i in range(n_neg)]
    labels = [1] * n_pos + [0] * n_neg

    frame = pd.DataFrame(
        {
            "uid": [str(i) for i in range(N_ROWS)],
            "subject": subjects,
            "priority": labels,
        }
    )
    frame = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)

    for name in FEATURE_COLUMNS:
        if name == "urgency_keyword_count":
            frame[name] = (frame["priority"] * 2).astype("int64")
        else:
            frame[name] = rng.integers(0, 20, size=N_ROWS).astype("int64")

    return frame


@pytest.fixture
def fitted_vectorizer(feature_frame: pd.DataFrame):
    """Vectorizer fitted on the frame's subjects (stand-in for the train split)."""
    return fit_subject_vectorizer(feature_frame["subject"], max_features=100)


# --------------------------------------------------------------------------
# stratified_split
# --------------------------------------------------------------------------


def test_stratified_split_proportions(feature_frame: pd.DataFrame) -> None:
    train_df, val_df, test_df = stratified_split(feature_frame)
    assert len(train_df) == pytest.approx(0.70 * N_ROWS, abs=2)
    assert len(val_df) == pytest.approx(0.15 * N_ROWS, abs=2)
    assert len(test_df) == pytest.approx(0.15 * N_ROWS, abs=2)
    assert len(train_df) + len(val_df) + len(test_df) == N_ROWS


def test_stratified_split_preserves_class_balance(feature_frame: pd.DataFrame) -> None:
    train_df, val_df, test_df = stratified_split(feature_frame)
    total_pos = int((feature_frame["priority"] == 1).sum())
    assert int((train_df["priority"] == 1).sum()) == pytest.approx(
        0.70 * total_pos, abs=1
    )
    assert int((val_df["priority"] == 1).sum()) + int(
        (test_df["priority"] == 1).sum()
    ) == pytest.approx(0.30 * total_pos, abs=1)


def test_stratified_split_is_disjoint(feature_frame: pd.DataFrame) -> None:
    train_df, val_df, test_df = stratified_split(feature_frame)
    uids = [set(df["uid"]) for df in (train_df, val_df, test_df)]
    assert uids[0] & uids[1] == set()
    assert uids[0] & uids[2] == set()
    assert uids[1] & uids[2] == set()


def test_stratified_split_is_reproducible(feature_frame: pd.DataFrame) -> None:
    first = stratified_split(feature_frame, seed=SPLIT_SEED)
    second = stratified_split(feature_frame, seed=SPLIT_SEED)
    for left, right in zip(first, second, strict=True):
        assert left["uid"].tolist() == right["uid"].tolist()


def test_stratified_split_seed_changes_split(feature_frame: pd.DataFrame) -> None:
    first = stratified_split(feature_frame, seed=SPLIT_SEED)[0]
    other = stratified_split(feature_frame, seed=SPLIT_SEED + 1)[0]
    assert first["uid"].tolist() != other["uid"].tolist()


def test_stratified_split_rejects_missing_target(feature_frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        stratified_split(feature_frame.drop(columns=["priority"]))


def test_stratified_split_rejects_bad_fractions(
    feature_frame: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        stratified_split(feature_frame, fractions=(0.6, 0.2, 0.3))


def test_stratified_split_rejects_nonpositive_fractions(
    feature_frame: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        stratified_split(feature_frame, fractions=(0.0, 0.5, 0.5))


# --------------------------------------------------------------------------
# build_matrices
# --------------------------------------------------------------------------


def test_build_matrices_appends_tfidf(
    feature_frame: pd.DataFrame, fitted_vectorizer
) -> None:
    X = build_matrices(feature_frame, vectorizer=fitted_vectorizer, use_tfidf=True)
    expected = len(FEATURE_COLUMNS) + len(fitted_vectorizer.get_feature_names_out())
    assert X.shape == (N_ROWS, expected)


def test_build_matrices_is_sparse_never_dense(
    feature_frame: pd.DataFrame, fitted_vectorizer
) -> None:
    X = build_matrices(feature_frame, vectorizer=fitted_vectorizer)
    assert issparse(X)
    assert isinstance(X, csr_matrix)


def test_build_matrices_without_tfidf(feature_frame: pd.DataFrame) -> None:
    X = build_matrices(feature_frame, use_tfidf=False)
    assert X.shape == (N_ROWS, len(FEATURE_COLUMNS))


def test_build_matrices_is_deterministic(
    feature_frame: pd.DataFrame, fitted_vectorizer
) -> None:
    first = build_matrices(feature_frame, vectorizer=fitted_vectorizer)
    second = build_matrices(feature_frame, vectorizer=fitted_vectorizer)
    assert (first != second).nnz == 0


def test_build_matrices_rejects_missing_column(feature_frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="sender_frequency"):
        build_matrices(
            feature_frame.drop(columns=["sender_frequency"]), use_tfidf=False
        )


def test_build_matrices_requires_vectorizer(feature_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="requires a fitted vectorizer"):
        build_matrices(feature_frame, use_tfidf=True)


# --------------------------------------------------------------------------
# build_model
# --------------------------------------------------------------------------


def test_build_model_uses_weight_derived_from_training_labels() -> None:
    """Hand-made training labels have three negatives per positive."""
    train = pd.DataFrame({"priority": [0, 0, 1, 0]})
    weight = scale_pos_weight_from_labels(train["priority"])
    assert weight == 3.0
    assert build_model(scale_pos_weight=weight).get_params()["scale_pos_weight"] == 3.0


@pytest.mark.parametrize("labels", [[], [0, 0], [1, 1], [0, 2]])
def test_scale_pos_weight_rejects_undefined_training_ratio(labels: list[int]) -> None:
    with pytest.raises(ValueError, match="scale_pos_weight"):
        scale_pos_weight_from_labels(np.array(labels))


def test_build_model_does_not_double_count_with_class_weight() -> None:
    params = build_model().get_params()
    assert params.get("class_weight") is None


def test_build_model_scale_pos_weight_is_configurable() -> None:
    assert build_model(scale_pos_weight=99.0).get_params()["scale_pos_weight"] == 99.0


# --------------------------------------------------------------------------
# train_model
# --------------------------------------------------------------------------


def test_train_model_returns_fitted_model_and_iteration(
    feature_frame: pd.DataFrame, fitted_vectorizer
) -> None:
    train_df, val_df, _ = stratified_split(feature_frame)
    X_train = build_matrices(train_df, vectorizer=fitted_vectorizer)
    X_val = build_matrices(val_df, vectorizer=fitted_vectorizer)

    model, best_iteration = train_model(
        build_model(n_estimators=50),
        X_train,
        train_df["priority"],
        X_val,
        val_df["priority"],
        stopping_rounds=10,
    )

    assert best_iteration >= 1
    assert hasattr(model, "booster_")
    assert len(model.predict(X_val)) == len(val_df)


def test_train_model_early_stops_on_class_one_f1(
    feature_frame: pd.DataFrame, fitted_vectorizer
) -> None:
    """Regression lock: early stopping must track class-1 F1, not logloss.

    LightGBM has no built-in "f1" metric, so `eval_metric="f1"` is silently
    ignored and falls back to binary_logloss. This asserts the custom metric is
    the ONLY one reported, which is what makes early stopping track it.
    """
    train_df, val_df, _ = stratified_split(feature_frame)
    X_train = build_matrices(train_df, vectorizer=fitted_vectorizer)
    X_val = build_matrices(val_df, vectorizer=fitted_vectorizer)

    model, _ = train_model(
        build_model(n_estimators=50),
        X_train,
        train_df["priority"],
        X_val,
        val_df["priority"],
        stopping_rounds=10,
    )

    reported = {name for series in model.evals_result_.values() for name in series}
    assert reported == {"f1_pos"}
    assert "binary_logloss" not in reported


def test_early_stopping_f1_metric_name_and_direction() -> None:
    name, value, higher_is_better = early_stopping_f1(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.7, 0.8])
    )
    assert name == "f1_pos"
    assert higher_is_better is True
    assert value == pytest.approx(1.0)


def test_early_stopping_f1_is_zero_when_nothing_predicted_positive() -> None:
    name, value, higher_is_better = early_stopping_f1(
        np.array([0, 1, 1]), np.array([0.1, 0.2, 0.3])
    )
    assert value == 0.0
    assert higher_is_better is True


def test_build_model_disables_builtin_metrics() -> None:
    """binary_logloss must not shadow the custom F1 during early stopping."""
    assert build_model().get_params()["metric"] == "None"


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------


class _FixedPredictor:
    """Duck-typed stand-in returning hard-coded predictions, for exact metrics."""

    def __init__(self, y_pred: list[int], y_proba: list[float]) -> None:
        self._y_pred = np.asarray(y_pred)
        self._y_proba = np.asarray(y_proba)

    def predict(self, X: csr_matrix) -> np.ndarray:
        return self._y_pred

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        return np.column_stack([1 - self._y_proba, self._y_proba])


# tp=2, fp=1, fn=2, tn=3 -> P=2/3, R=1/2, F1=4/7, acc=5/8
_Y_TRUE = [0, 0, 0, 1, 1, 1, 1, 0]
_Y_PRED = [0, 0, 1, 1, 1, 0, 0, 0]
_Y_PROBA = [0.1, 0.2, 0.6, 0.7, 0.8, 0.4, 0.3, 0.15]


def test_evaluate_exact_class_one_metrics() -> None:
    X = csr_matrix((8, 1))
    scores = evaluate(_FixedPredictor(_Y_PRED, _Y_PROBA), X, _Y_TRUE)

    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == pytest.approx(0.5)
    assert scores["f1"] == pytest.approx(4 / 7)
    assert scores["accuracy"] == pytest.approx(5 / 8)


def test_evaluate_confusion_matrix_cells() -> None:
    X = csr_matrix((8, 1))
    scores = evaluate(_FixedPredictor(_Y_PRED, _Y_PROBA), X, _Y_TRUE)
    assert scores["true_positive"] == 2
    assert scores["false_positive"] == 1
    assert scores["false_negative"] == 2
    assert scores["true_negative"] == 3


def test_evaluate_includes_roc_auc_when_both_classes_present() -> None:
    """14 of 16 (positive, negative) pairs are correctly ordered -> 0.875."""
    X = csr_matrix((8, 1))
    scores = evaluate(_FixedPredictor(_Y_PRED, _Y_PROBA), X, _Y_TRUE)
    assert scores["roc_auc"] == pytest.approx(14 / 16)


def test_evaluate_omits_roc_auc_for_single_class() -> None:
    X = csr_matrix((3, 1))
    scores = evaluate(_FixedPredictor([0, 0, 0], [0.1, 0.2, 0.3]), X, [0, 0, 0])
    assert "roc_auc" not in scores
    assert scores["recall"] == 0.0


def test_evaluate_primary_metric_is_f1() -> None:
    """Accuracy is reported but is never the headline for the imbalanced target."""
    X = csr_matrix((8, 1))
    scores = evaluate(_FixedPredictor(_Y_PRED, _Y_PROBA), X, _Y_TRUE)
    assert scores["accuracy"] > scores["f1"]
    assert set(scores) >= {"f1", "precision", "recall"}


# --------------------------------------------------------------------------
# feature_importance_frame
# --------------------------------------------------------------------------


def test_feature_importance_frame_sorted_by_gain(
    feature_frame: pd.DataFrame, fitted_vectorizer
) -> None:
    train_df, val_df, _ = stratified_split(feature_frame)
    X_train = build_matrices(train_df, vectorizer=fitted_vectorizer)
    X_val = build_matrices(val_df, vectorizer=fitted_vectorizer)
    model, _ = train_model(
        build_model(n_estimators=20),
        X_train,
        train_df["priority"],
        X_val,
        val_df["priority"],
        stopping_rounds=5,
    )

    frame = feature_importance_frame(model, vectorizer=fitted_vectorizer, top_n=None)
    assert frame["gain"].is_monotonic_decreasing
    assert np.isfinite(frame["gain"]).all()
    assert set(frame.columns) == {"feature", "gain", "gain_pct"}


def test_feature_importance_frame_has_one_row_per_feature(
    feature_frame: pd.DataFrame, fitted_vectorizer
) -> None:
    train_df, val_df, _ = stratified_split(feature_frame)
    X_train = build_matrices(train_df, vectorizer=fitted_vectorizer)
    X_val = build_matrices(val_df, vectorizer=fitted_vectorizer)
    model, _ = train_model(
        build_model(n_estimators=10),
        X_train,
        train_df["priority"],
        X_val,
        val_df["priority"],
        stopping_rounds=5,
    )

    frame = feature_importance_frame(model, vectorizer=fitted_vectorizer, top_n=None)
    expected = len(FEATURE_COLUMNS) + len(fitted_vectorizer.get_feature_names_out())
    assert len(frame) == expected
    assert frame["feature"].nunique() == expected
    assert frame["gain_pct"].sum() == pytest.approx(100.0, abs=1e-6)


def test_feature_importance_frame_respects_top_n(
    feature_frame: pd.DataFrame, fitted_vectorizer
) -> None:
    train_df, val_df, _ = stratified_split(feature_frame)
    X_train = build_matrices(train_df, vectorizer=fitted_vectorizer)
    X_val = build_matrices(val_df, vectorizer=fitted_vectorizer)
    model, _ = train_model(
        build_model(n_estimators=10),
        X_train,
        train_df["priority"],
        X_val,
        val_df["priority"],
        stopping_rounds=5,
    )
    assert (
        len(feature_importance_frame(model, vectorizer=fitted_vectorizer, top_n=5)) == 5
    )


def test_feature_importance_frame_detects_name_mismatch(
    feature_frame: pd.DataFrame, fitted_vectorizer
) -> None:
    train_df, val_df, _ = stratified_split(feature_frame)
    X_train = build_matrices(train_df, vectorizer=fitted_vectorizer)
    X_val = build_matrices(val_df, vectorizer=fitted_vectorizer)
    model, _ = train_model(
        build_model(n_estimators=10),
        X_train,
        train_df["priority"],
        X_val,
        val_df["priority"],
        stopping_rounds=5,
    )
    with pytest.raises(ValueError, match="does not match"):
        feature_importance_frame(
            model, vectorizer=fitted_vectorizer, feature_columns=BEHAVIORAL_FEATURES
        )


def test_feature_importance_frame_requires_vectorizer(
    feature_frame: pd.DataFrame, fitted_vectorizer
) -> None:
    train_df, val_df, _ = stratified_split(feature_frame)
    X_train = build_matrices(train_df, vectorizer=fitted_vectorizer)
    X_val = build_matrices(val_df, vectorizer=fitted_vectorizer)
    model, _ = train_model(
        build_model(n_estimators=10),
        X_train,
        train_df["priority"],
        X_val,
        val_df["priority"],
        stopping_rounds=5,
    )
    with pytest.raises(ValueError, match="requires a fitted vectorizer"):
        feature_importance_frame(model, use_tfidf=True)


# --------------------------------------------------------------------------
# plot_feature_importance
# --------------------------------------------------------------------------


def test_plot_feature_importance_writes_png(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "feature": ["urgency_keyword_count", "tfidf::urgent", "sender_frequency"],
            "gain": [120.0, 80.0, 40.0],
            "gain_pct": [50.0, 33.3, 16.7],
        }
    )
    output = plot_feature_importance(frame, tmp_path, title="with-keyword-features")
    assert output.is_file()
    assert output.read_bytes().startswith(b"\x89PNG")
    assert output.name == "feature_importance_with-keyword-features.png"


def test_plot_feature_importance_creates_output_dir(tmp_path: Path) -> None:
    frame = pd.DataFrame({"feature": ["a"], "gain": [1.0], "gain_pct": [100.0]})
    target = tmp_path / "nested" / "dir"
    output = plot_feature_importance(frame, target, title="run")
    assert target.is_dir()
    assert output.is_file()


# --------------------------------------------------------------------------
# variants
# --------------------------------------------------------------------------


def test_variants_cover_both_documented_runs() -> None:
    assert [v.name for v in VARIANTS] == [
        "with-keyword-features",
        "behavioral-only",
    ]


def test_with_keyword_variant_uses_all_features_and_tfidf() -> None:
    variant = variant_by_name("with-keyword-features")
    assert variant.feature_columns == FEATURE_COLUMNS
    assert variant.use_tfidf is True


def test_behavioral_variant_excludes_keyword_and_tfidf() -> None:
    """Guards the label-leakage escape hatch agreed for the second run."""
    variant = variant_by_name("behavioral-only")
    assert "urgency_keyword_count" not in variant.feature_columns
    assert variant.use_tfidf is False
    assert variant.feature_columns == BEHAVIORAL_FEATURES
    assert len(variant.feature_columns) == 7


def test_variant_by_name_rejects_unknown() -> None:
    with pytest.raises(KeyError):
        variant_by_name("does-not-exist")
