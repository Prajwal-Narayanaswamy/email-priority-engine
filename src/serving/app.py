"""FastAPI app: /health, /predict, /model-info for the ADR-013 reference model."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from src.serving import config
from src.serving.lookup import load_sender_lookup
from src.serving.predictor import (
    CAVEATS,
    SENDER_FREQUENCY_NOTE,
    Predictor,
    load_model,
    load_threshold,
)
from src.serving.schemas import EmailIn, HealthOut, ModelInfoOut, PredictionOut
from src.training.train import BEHAVIORAL_FEATURES

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model, threshold and sender lookup once at startup."""
    if getattr(app.state, "predictor", None) is None:
        app.state.predictor = Predictor(
            model=load_model(config.MODEL_RUN_ID),
            threshold=load_threshold(config.EVAL_RUN_ID),
            lookup=load_sender_lookup(config.SENDER_LOOKUP_PATH),
            model_run_id=config.MODEL_RUN_ID,
            eval_run_id=config.EVAL_RUN_ID,
        )
        logger.info("Serving model %s", config.MODEL_RUN_ID)
    yield


app = FastAPI(title="Email Priority Reference API", lifespan=lifespan)


def _predictor(request: Request) -> Predictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return predictor


@app.get("/health", response_model=HealthOut)
def health(request: Request) -> HealthOut:
    """Liveness probe with the loaded model run ID."""
    return HealthOut(status="ok", model_run_id=_predictor(request).model_run_id)


@app.post("/predict", response_model=PredictionOut)
def predict(email: EmailIn, request: Request) -> PredictionOut:
    """Score one email with the reference model."""
    predictor = _predictor(request)
    probability, flagged, seen, features = predictor.predict(
        email.sender, email.subject, email.date
    )
    return PredictionOut(
        probability=probability,
        flagged=flagged,
        threshold=predictor.threshold,
        sender_seen_in_training=seen,
        features=features,
        model_run_id=predictor.model_run_id,
        caveats=CAVEATS,
    )


@app.get("/model-info", response_model=ModelInfoOut)
def model_info(request: Request) -> ModelInfoOut:
    """Describe the loaded model, threshold, features and limitations."""
    predictor = _predictor(request)
    return ModelInfoOut(
        model_run_id=predictor.model_run_id,
        eval_run_id=predictor.eval_run_id,
        threshold=predictor.threshold,
        feature_columns=list(BEHAVIORAL_FEATURES),
        sender_lookup_size=len(predictor.lookup),
        caveats=CAVEATS,
        sender_frequency_note=SENDER_FREQUENCY_NOTE,
    )
