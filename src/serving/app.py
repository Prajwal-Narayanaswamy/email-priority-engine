"""FastAPI app: /health, /predict, /model-info for the ADR-013 reference model."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from mlflow import MlflowClient

from src.serving import config
from src.serving.lookup import load_lookup
from src.serving.predictor import (
    CAVEATS,
    PREDICT_NOTE,
    SENDER_FREQUENCY_NOTE,
    Predictor,
)
from src.serving.registry import (
    ServingStartupError,
    configure_mlflow,
    fetch_serving_metadata,
    load_model,
)
from src.serving.schemas import EmailIn, HealthOut, ModelInfoOut, PredictionOut
from src.training.train import BEHAVIORAL_FEATURES

logger = logging.getLogger(__name__)


def build_predictor() -> Predictor:
    """Resolve MODEL_URI, verify the lookup against its tag and load the model.

    Raises:
        ServingStartupError: on any registry, tag, threshold or lookup mismatch.
        FileNotFoundError: if the lookup file is missing.
        LookupFormatError: if the lookup has no or a malformed header.
    """
    configure_mlflow(config.TRACKING_URI)
    client = MlflowClient(tracking_uri=config.TRACKING_URI)
    metadata = fetch_serving_metadata(config.MODEL_URI, client)
    lookup_file = load_lookup(config.SENDER_LOOKUP_PATH)
    if lookup_file.file_sha256 != metadata.lookup_sha256:
        raise ServingStartupError(
            f"lookup file sha256 {lookup_file.file_sha256} does not match the "
            f"registered lookup_sha256 tag {metadata.lookup_sha256}; rebuild or "
            "re-register"
        )
    model = load_model(metadata, config.TRACKING_URI)
    logger.info(
        "Serving %s version %s (threshold %s)",
        metadata.model_name,
        metadata.version,
        metadata.threshold,
    )
    return Predictor(model, metadata, lookup_file.senders)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the predictor once at startup; any failure aborts startup."""
    if getattr(app.state, "predictor", None) is None:
        app.state.predictor = build_predictor()
    yield


app = FastAPI(title="Email Priority Reference API", lifespan=lifespan)


def _predictor(request: Request) -> Predictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return predictor


@app.get("/health", response_model=HealthOut)
def health(request: Request) -> HealthOut:
    """Liveness probe with the loaded model's provenance."""
    predictor = _predictor(request)
    return HealthOut(
        status="ok",
        model_run_id=predictor.model_run_id,
        model_version=predictor.model_version,
    )


@app.post("/predict", response_model=PredictionOut)
def predict(email: EmailIn, request: Request) -> PredictionOut:
    """Score one email with the reference model."""
    predictor = _predictor(request)
    score, flagged, seen, features = predictor.predict(
        email.sender, email.subject, email.date
    )
    return PredictionOut(
        score=score,
        flagged=flagged,
        threshold=predictor.threshold,
        sender_seen_in_training=seen,
        features=features,
        model_run_id=predictor.model_run_id,
        model_version=predictor.model_version,
        note=PREDICT_NOTE,
    )


@app.get("/model-info", response_model=ModelInfoOut)
def model_info(request: Request) -> ModelInfoOut:
    """Describe the loaded model version, tags, threshold and limitations."""
    predictor = _predictor(request)
    meta = predictor.metadata
    return ModelInfoOut(
        model_name=meta.model_name,
        model_version=meta.version,
        alias=meta.alias,
        model_uri=meta.model_uri,
        model_run_id=predictor.model_run_id,
        threshold=meta.threshold,
        threshold_source_run=meta.tags["threshold_source_run"],
        lookup_sha256=meta.lookup_sha256,
        tags=meta.tags,
        feature_columns=list(BEHAVIORAL_FEATURES),
        sender_lookup_size=len(predictor.lookup),
        caveats=CAVEATS,
        sender_frequency_note=SENDER_FREQUENCY_NOTE,
    )
