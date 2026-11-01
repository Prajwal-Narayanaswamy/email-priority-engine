"""Model loading and single-email prediction (framework-free, testable)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.features.features import (
    feature_day_of_week,
    feature_has_question_mark,
    feature_hour_of_day,
    feature_sender_is_leadership,
    feature_subject_length,
    feature_thread_depth,
)
from src.serving.lookup import sender_frequency_for
from src.serving.registry import ServingMetadata
from src.training.train import BEHAVIORAL_FEATURES

logger = logging.getLogger(__name__)

CAVEATS: str = (
    "Reference model per ADR-013; not a genuine urgency predictor. "
    "Labels are heuristic (ADR-005). "
    "The score is uncalibrated and inflated by scale_pos_weight; it is not a "
    "probability. "
    "On the held-out test split, the flagged rows were exactly the "
    "leadership-sender rows (60 of 60; test F1 about 0.42, the same as the "
    "leadership-only rule; non-leadership F1 0). That is an empirical property "
    "of that split, not a guarantee: a leadership sender's email can score "
    "below the threshold."
)
PREDICT_NOTE: str = "Reference model; see /model-info for caveats."
SENDER_FREQUENCY_NOTE: str = (
    "sender_frequency is a snapshot of training-data counts and does not "
    "update with new emails. Unseen senders are assigned 1 (the lowest value "
    "seen in training); null/empty senders get 0."
)


def build_feature_row(
    sender: str | None, subject: str | None, date: str | None, lookup: dict[str, int]
) -> tuple[pd.DataFrame, bool]:
    """Derive the 7 behavioral features for one email.

    Reuses the training feature functions; only sender_frequency comes from
    the saved lookup. Returns (one-row frame in model column order, seen).
    """
    frame = pd.DataFrame({"from": [sender], "subject": [subject], "date": [date]})
    frequency, seen = sender_frequency_for(sender, lookup)
    frame["sender_frequency"] = frequency
    frame["sender_is_leadership"] = feature_sender_is_leadership(frame)
    frame["hour_of_day"] = feature_hour_of_day(frame)
    frame["day_of_week"] = feature_day_of_week(frame)
    frame["subject_length"] = feature_subject_length(frame)
    frame["has_question_mark"] = feature_has_question_mark(frame)
    frame["thread_depth"] = feature_thread_depth(frame)
    return frame[list(BEHAVIORAL_FEATURES)], seen


class Predictor:
    """Holds the loaded model, threshold and sender lookup for serving."""

    def __init__(
        self,
        model: Any,
        metadata: ServingMetadata,
        lookup: dict[str, int],
    ) -> None:
        self.model = model
        self.metadata = metadata
        self.lookup = lookup

    @property
    def threshold(self) -> float:
        """Validation-tuned threshold read from the registered version's tags."""
        return self.metadata.threshold

    @property
    def model_run_id(self) -> str:
        """Training run that produced the served model (provenance)."""
        return self.metadata.tags["source_run_id"]

    @property
    def model_version(self) -> str:
        """Registry version number being served."""
        return self.metadata.version

    def predict(
        self, sender: str | None, subject: str | None, date: str | None
    ) -> tuple[float, bool, bool, dict[str, float]]:
        """Return (score, flagged, sender_seen, features).

        The score is the model's uncalibrated class-1 output (inflated by
        scale_pos_weight), compared against the validation-tuned threshold.
        """
        row, seen = build_feature_row(sender, subject, date, self.lookup)
        score = float(self.model.predict_proba(row)[0, 1])
        features = {k: float(v) for k, v in row.iloc[0].items()}
        return score, score >= self.threshold, seen, features
