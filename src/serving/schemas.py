"""Pydantic request/response schemas for the serving API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EmailIn(BaseModel):
    """One raw email, matching the ingest columns the features derive from."""

    model_config = ConfigDict(populate_by_name=True)

    sender: str | None = Field(default=None, alias="from")
    subject: str | None = None
    date: str | None = Field(
        default=None, description="Date header; unparseable -> hour/day = -1"
    )


class PredictionOut(BaseModel):
    """Prediction for one email, always with the model's caveats attached."""

    probability: float
    flagged: bool
    threshold: float
    sender_seen_in_training: bool
    features: dict[str, float]
    model_run_id: str
    caveats: str


class HealthOut(BaseModel):
    """Liveness response."""

    status: str
    model_run_id: str


class ModelInfoOut(BaseModel):
    """Description of the loaded reference model and its limitations."""

    model_run_id: str
    eval_run_id: str
    threshold: float
    feature_columns: list[str]
    sender_lookup_size: int
    caveats: str
    sender_frequency_note: str
