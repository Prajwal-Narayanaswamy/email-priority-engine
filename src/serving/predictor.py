"""Model loading and single-email prediction (framework-free, testable)."""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import mlflow.sklearn  # noqa: F401  (registers mlflow.sklearn.load_model)
import pandas as pd
from mlflow import MlflowClient

from src.features.features import (
    feature_day_of_week,
    feature_has_question_mark,
    feature_hour_of_day,
    feature_sender_is_leadership,
    feature_subject_length,
    feature_thread_depth,
)
from src.serving import config
from src.serving.lookup import sender_frequency_for
from src.training.train import BEHAVIORAL_FEATURES, LIMITATION_NOTE

logger = logging.getLogger(__name__)

CAVEATS: str = (
    "Reference model (ADR-013). Labels are heuristic (ADR-005). At the "
    "operating threshold the model flags exactly the leadership-sender rows, "
    "identical to the leadership-only rule (test F1 ~0.42; non-leadership F1 "
    "0). Not a genuine urgency predictor. " + LIMITATION_NOTE
)
SENDER_FREQUENCY_NOTE: str = (
    "sender_frequency is a snapshot of training-data counts and does not "
    "update with new emails. Unseen senders are assigned 1 (the lowest value "
    "seen in training); null/empty senders get 0."
)


def load_model(run_id: str) -> Any:
    """Load the single logged 'model' LoggedModel of a training run."""
    mlflow.set_tracking_uri(config.TRACKING_URI)
    client = MlflowClient(tracking_uri=config.TRACKING_URI)
    experiment = client.get_experiment_by_name(config.EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"experiment {config.EXPERIMENT_NAME!r} not found")
    candidates = [
        m
        for m in client.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"source_run_id = '{run_id}'",
        )
        if m.name == "model"
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected one 'model' for run {run_id}, got {len(candidates)}"
        )
    logger.info("Loading model %s from run %s", candidates[0].model_id, run_id)
    return mlflow.sklearn.load_model(f"models:/{candidates[0].model_id}")


def load_threshold(eval_run_id: str) -> float:
    """Read the validation-selected threshold logged by the evaluation run."""
    client = MlflowClient(tracking_uri=config.TRACKING_URI)
    return float(client.get_run(eval_run_id).data.params["tuned_threshold"])


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
        threshold: float,
        lookup: dict[str, int],
        model_run_id: str,
        eval_run_id: str,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.lookup = lookup
        self.model_run_id = model_run_id
        self.eval_run_id = eval_run_id

    def predict(
        self, sender: str | None, subject: str | None, date: str | None
    ) -> tuple[float, bool, bool, dict[str, float]]:
        """Return (probability, flagged, sender_seen, features)."""
        row, seen = build_feature_row(sender, subject, date, self.lookup)
        probability = float(self.model.predict_proba(row)[0, 1])
        features = {k: float(v) for k, v in row.iloc[0].items()}
        return probability, probability >= self.threshold, seen, features
