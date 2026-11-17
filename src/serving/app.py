"""FastAPI app: /health, /predict, /model-info for the ADR-013 reference model."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from src.features.columns import BEHAVIORAL_FEATURES
from src.serving import bundle, config, versions
from src.serving.lookup import load_lookup
from src.serving.metadata import ServingMetadata, ServingStartupError
from src.serving.predictor import (
    CAVEATS,
    PREDICT_NOTE,
    SENDER_FREQUENCY_NOTE,
    Predictor,
)
from src.serving.schemas import EmailIn, HealthOut, ModelInfoOut, PredictionOut

logger = logging.getLogger(__name__)

MODEL_SOURCES: tuple[str, ...] = ("registry", "bundle")


def _check_lookup(metadata: ServingMetadata, path: Path) -> dict[str, int]:
    """Load the lookup and require its file sha256 to equal the metadata's tag."""
    lookup_file = load_lookup(path)
    if lookup_file.file_sha256 != metadata.lookup_sha256:
        raise ServingStartupError(
            f"lookup file sha256 {lookup_file.file_sha256} does not match the "
            f"registered lookup_sha256 tag {metadata.lookup_sha256}; rebuild or "
            "re-register"
        )
    return lookup_file.senders


def _from_registry() -> tuple[ServingMetadata, dict[str, int], Any]:
    # Imported here so a bundle-only image does not need mlflow.
    from src.serving import registry

    registry.configure_mlflow(config.TRACKING_URI)
    client = registry.MlflowClient(tracking_uri=config.TRACKING_URI)
    metadata = registry.fetch_serving_metadata(config.MODEL_URI, client)
    lookup = _check_lookup(metadata, config.SENDER_LOOKUP_PATH)
    return metadata, lookup, registry.load_model(metadata, config.TRACKING_URI)


def _from_bundle() -> tuple[ServingMetadata, dict[str, int], Any]:
    metadata = bundle.load_bundle_metadata(config.BUNDLE_DIR)
    lookup = _check_lookup(metadata, config.BUNDLE_DIR / bundle.LOOKUP_FILE)
    return metadata, lookup, bundle.load_bundle_model(config.BUNDLE_DIR)


def build_predictor() -> Predictor:
    """Load model, metadata and lookup from the explicit MODEL_SOURCE.

    There is no default and no fallback: an unset or unknown MODEL_SOURCE, or
    any failure in the chosen source, stops startup.

    Raises:
        ServingStartupError: on a bad MODEL_SOURCE or any registry, bundle, tag,
            threshold or lookup-hash problem.
        FileNotFoundError: if the lookup file is missing.
        LookupFormatError: if the lookup has no or a malformed header.
    """
    source = config.MODEL_SOURCE
    if source == "registry":
        metadata, lookup, model = _from_registry()
    elif source == "bundle":
        metadata, lookup, model = _from_bundle()
    else:
        raise ServingStartupError(
            f"MODEL_SOURCE must be one of {list(MODEL_SOURCES)} "
            f"(got {source!r}); there is no default and no fallback"
        )
    versions.warn_on_mismatch(versions.runtime_versions(), metadata.bundle_versions)
    logger.info(
        "Serving %s version %s from %s (threshold %s)",
        metadata.model_name,
        metadata.version,
        source,
        metadata.threshold,
    )
    return Predictor(model, metadata, lookup)


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
        runtime_versions=versions.runtime_versions(),
        bundle_versions=meta.bundle_versions,
        caveats=CAVEATS,
        sender_frequency_note=SENDER_FREQUENCY_NOTE,
    )
