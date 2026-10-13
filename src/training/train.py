"""Model training for the email priority classifier.

Pure functions — no Prefect, no MLflow. Both live in the orchestration layer
(src/tasks/train_tasks.py), matching the pattern set by ingest.py and
features.py (ADR-008).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.features.features import FEATURE_COLUMNS, transform_subjects

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

TARGET: str = "priority"
SUBJECT: str = "subject"

SPLIT_SEED: int = 42
SPLIT_FRACTIONS: tuple[float, float, float] = (0.70, 0.15, 0.15)

# The label rule (ADR-005) reads urgency keywords out of the subject, and the
# subject is also a feature source. `behavioral-only` exists to measure how much
# of the score is the model re-deriving the heuristic rather than finding real
# priority signal. See LIMITATION_NOTE.
LIMITATION_NOTE: str = (
    "Labels are heuristic (ADR-005): priority=1 when the sender is in the "
    "leadership list OR the subject contains an urgency keyword. Because "
    "urgency_keyword_count and the subject TF-IDF matrix encode that same "
    "signal, the with-keyword-features run can largely re-derive the label "
    "rule instead of learning genuine priority. A high F1 is not evidence of "
    "real-world priority prediction. Compare against behavioral-only, which "
    "drops urgency_keyword_count and subject TF-IDF. Second known limitation: "
    "sender_frequency is computed on the full dataset including the test "
    "split (minor leakage, accepted for v1)."
)

#: Tabular features that do not encode the ADR-005 label rule.
BEHAVIORAL_FEATURES: tuple[str, ...] = tuple(
    c for c in FEATURE_COLUMNS if c != "urgency_keyword_count"
)


class VariantSpec(NamedTuple):
    """One training configuration, run and logged as its own MLflow run."""

    name: str
    feature_columns: tuple[str, ...]
    use_tfidf: bool
    description: str


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        name="with-keyword-features",
        feature_columns=FEATURE_COLUMNS,
        use_tfidf=True,
        description="All 8 tabular features + subject TF-IDF (top 100).",
    ),
    VariantSpec(
        name="behavioral-only",
        feature_columns=BEHAVIORAL_FEATURES,
        use_tfidf=False,
        description=(
            "7 behavioral/structural tabular features only; excludes "
            "urgency_keyword_count and subject TF-IDF to escape label leakage."
        ),
    ),
)


def variant_by_name(name: str) -> VariantSpec:
    """Look up a training variant by name.

    Raises:
        KeyError: if no variant carries that name.
    """
    for variant in VARIANTS:
        if variant.name == name:
            return variant
    raise KeyError(
        f"unknown variant {name!r}; expected one of {[v.name for v in VARIANTS]}"
    )


def stratified_split(
    df: pd.DataFrame,
    *,
    seed: int = SPLIT_SEED,
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test, stratified on the target column.

    70/15/15 by default. The validation split drives early stopping; the test
    split is held back for final reporting only.

    Args:
        df: Feature frame containing at least the target column.
        seed: Seed shared by both `train_test_split` calls, so the split is
            reproducible and identical across training variants.
        fractions: (train, val, test) proportions; must sum to 1.

    Returns:
        (train_df, val_df, test_df) as disjoint frames.

    Raises:
        KeyError: if the target column is absent.
        ValueError: if fractions do not sum to 1 or a split would be empty.
    """
    if TARGET not in df.columns:
        raise KeyError(f"column {TARGET!r} not present in frame")

    train_frac, val_frac, test_frac = fractions
    if not math.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1.0, got {sum(fractions)}")
    if min(fractions) <= 0:
        raise ValueError(f"all fractions must be positive, got {fractions}")

    holdout = val_frac + test_frac
    train_df, remainder = train_test_split(
        df,
        test_size=holdout,
        random_state=seed,
        stratify=df[TARGET],
    )
    val_df, test_df = train_test_split(
        remainder,
        test_size=test_frac / holdout,
        random_state=seed,
        stratify=remainder[TARGET],
    )
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError(
            f"degenerate split for {len(df)} rows with fractions {fractions}"
        )
    return train_df, val_df, test_df


def build_matrices(
    df: pd.DataFrame,
    *,
    vectorizer: TfidfVectorizer | None = None,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    use_tfidf: bool = True,
) -> csr_matrix:
    """Assemble the model matrix as a scipy CSR sparse matrix.

    Never returns a dense array: 517,401 x 108 float32 is ~200MB and would
    risk OOM on a 5.6GB host (ADR-008).

    Args:
        df: Feature frame.
        vectorizer: Fitted vectorizer, required when `use_tfidf` is True.
        feature_columns: Tabular feature columns to include, in order.
        use_tfidf: Append the subject TF-IDF block.

    Returns:
        CSR matrix of shape (len(df), len(feature_columns) + tfidf_width).

    Raises:
        KeyError: if a requested feature column is absent.
        ValueError: if `use_tfidf` is True but no vectorizer was supplied.
    """
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise KeyError(f"feature columns not present in frame: {missing}")

    tabular = csr_matrix(df[list(feature_columns)].to_numpy(dtype=np.float32))
    if not use_tfidf:
        return tabular
    if vectorizer is None:
        raise ValueError("use_tfidf=True requires a fitted vectorizer")
    tfidf = transform_subjects(vectorizer, df[SUBJECT])
    return hstack([tabular, tfidf], format="csr")


def scale_pos_weight_from_labels(y_train: pd.Series | np.ndarray) -> float:
    """Derive class imbalance from training labels only (ADR-012).

    Returns negatives / positives; reject empty, non-binary, or single-class
    labels rather than training with an undefined or zero class weight.
    """
    labels = np.asarray(y_train).ravel()
    if labels.size == 0 or not np.all(np.isin(labels, (0, 1))):
        raise ValueError("scale_pos_weight requires nonempty binary 0/1 labels")
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("scale_pos_weight requires both training classes")
    return float(negatives / positives)


def build_model(
    *,
    scale_pos_weight: float = 1.0,
    n_estimators: int = 2000,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    random_state: int = SPLIT_SEED,
    n_jobs: int = -1,
) -> LGBMClassifier:
    """Build the LightGBM binary classifier with an explicit imbalance weight.

    `scale_pos_weight` is the only imbalance lever — `class_weight` is left at
    its default because setting both double-counts the same correction.
    Training tasks supply the ratio derived from their training labels; the
    standalone builder defaults to neutral weighting, not a dataset constant.
    No feature scaling: trees are invariant to monotone feature transforms.

    `metric="None"` disables LightGBM's built-in metrics. They must be off, or
    `binary_logloss` is listed first and early stopping silently tracks logloss
    instead of the class-1 F1 this project optimises.
    """
    return LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=-1,
        metric="None",
    )


def early_stopping_f1(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[str, float, bool]:
    """Custom LightGBM eval metric scoring the positive (priority) class.

    LightGBM has no built-in F1 metric. Passing `eval_metric="f1"` is silently
    ignored and falls back to `binary_logloss`, which would make early stopping
    optimise log loss rather than the class-1 F1 used as the primary metric
    everywhere else in this project.

    Args:
        y_true: Validation labels.
        y_pred: Predicted probabilities for the positive class, as passed by
            LightGBM.

    Returns:
        The (name, value, higher_is_better) triple LightGBM requires from a
        custom metric.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_pred.shape[0] != y_true.shape[0]:
        y_pred = y_pred[: y_true.shape[0]]
    value = f1_score(y_true, (y_pred > 0.5).astype(int), pos_label=1, zero_division=0)
    return "f1_pos", float(value), True


def train_model(
    model: LGBMClassifier,
    X_train: csr_matrix,
    y_train: pd.Series | np.ndarray,
    X_val: csr_matrix,
    y_val: pd.Series | np.ndarray,
    *,
    stopping_rounds: int = 100,
) -> tuple[LGBMClassifier, int]:
    """Fit the model with early stopping on class-1 F1 from the validation split.

    Returns:
        (fitted_model, best_iteration)
    """
    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=np.asarray(y_val),
        eval_metric=early_stopping_f1,
        callbacks=[lgb.early_stopping(stopping_rounds, verbose=False)],
    )
    best_iteration = getattr(model, "best_iteration_", 0) or model.n_estimators
    return model, int(best_iteration)


def evaluate(
    model: LGBMClassifier, X: csr_matrix, y: pd.Series | np.ndarray
) -> dict[str, float | int]:
    """Score predictions with the positive class as the primary class.

    Returns:
        Dict with `f1`, `precision`, `recall` (all class-1 / positive class),
        `accuracy`, `roc_auc`, and the four confusion-matrix cells.
        `roc_auc` is omitted when the labels contain a single class.
    """
    y_true = np.asarray(y)
    y_pred = np.asarray(model.predict(X))

    scores: dict[str, float | int] = {
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    if np.unique(y_true).size > 1:
        y_proba = model.predict_proba(X)[:, 1]
        scores["roc_auc"] = float(roc_auc_score(y_true, y_proba))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    scores.update(
        {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }
    )
    return scores


def feature_importance_frame(
    model: LGBMClassifier,
    *,
    vectorizer: TfidfVectorizer | None = None,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    use_tfidf: bool = True,
    top_n: int | None = 20,
) -> pd.DataFrame:
    """Rank features by LightGBM split gain.

    Gain is used over default split count because it reflects how much each
    feature reduced loss, which is the more useful signal at this class ratio.

    Returns:
        Frame sorted by descending gain with columns feature, gain, gain_pct,
        truncated to `top_n` rows.

    Raises:
        ValueError: if the feature-name count does not match the fitted
            booster's importance count, or if `use_tfidf` without a vectorizer.
    """
    gain = np.asarray(
        model.booster_.feature_importance(importance_type="gain"), dtype=float
    )
    names = list(feature_columns)
    if use_tfidf:
        if vectorizer is None:
            raise ValueError("use_tfidf=True requires a fitted vectorizer")
        names += [f"tfidf::{term}" for term in vectorizer.get_feature_names_out()]

    if len(names) != len(gain):
        raise ValueError(
            f"feature name count ({len(names)}) does not match booster "
            f"importance count ({len(gain)})"
        )

    frame = pd.DataFrame({"feature": names, "gain": gain})
    total = float(frame["gain"].sum())
    frame["gain_pct"] = frame["gain"] / total * 100.0 if total > 0 else 0.0
    frame = frame.sort_values("gain", ascending=False).reset_index(drop=True)
    return frame if top_n is None else frame.head(top_n)


def plot_feature_importance(
    frame: pd.DataFrame, output_dir: Path, *, title: str, top_n: int = 20
) -> Path:
    """Render a horizontal gain bar chart of the top features to a PNG.

    Uses the Agg backend, so it works headless in WSL/Docker.

    Returns:
        Path to the written PNG. The file is named after the run variant.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    top = frame.head(top_n).iloc[::-1]

    figure, axes = plt.subplots(figsize=(10, 0.4 * len(top) + 1.5))
    axes.barh(top["feature"], top["gain_pct"], color="#3b6ea5")
    axes.set_xlabel("Gain importance (% of total)")
    axes.set_title(title)
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()

    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in title)
    output_path = output_dir / f"feature_importance_{slug}.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
