"""Registry access for serving: resolve MODEL_URI once, read tags, load the model.

Metadata retrieval is isolated in `fetch_serving_metadata`, which returns a
`ServingMetadata`; the bundle source (src/serving/bundle.py) returns the same
type. The dataclass and its helpers live in src/serving/metadata.py so the
bundle path does not import mlflow.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import mlflow
import mlflow.sklearn  # noqa: F401  (registers mlflow.sklearn.load_model)
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode

from src.serving.metadata import (
    ServingMetadata,
    ServingStartupError,
    check_required_tags,
    parse_threshold,
)

logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME: str = "email-priority"
CHAMPION_ALIAS: str = "champion"
#: ADR-013 reference: behavioral-only training run and its evaluation run.
REFERENCE_SOURCE_RUN_ID: str = "7189707b198f47748f89965712eee4bc"
REFERENCE_EVAL_RUN_ID: str = "559ccf6356494eecabb332c47eff2105"
_ALIAS_URI = re.compile(r"^models:/([^/@]+)@([^/@]+)$")
_VERSION_URI = re.compile(r"^models:/([^/@]+)/(\d+)$")


def parse_model_uri(model_uri: str) -> tuple[str, str | None, str | None]:
    """Split MODEL_URI into (name, alias, version); exactly one of the last two.

    Raises:
        ServingStartupError: if the URI is neither alias nor version form.
    """
    alias_match = _ALIAS_URI.match(model_uri)
    if alias_match:
        return alias_match.group(1), alias_match.group(2), None
    version_match = _VERSION_URI.match(model_uri)
    if version_match:
        return version_match.group(1), None, version_match.group(2)
    raise ServingStartupError(
        f"MODEL_URI {model_uri!r} must be models:/<name>@<alias> or "
        "models:/<name>/<version>"
    )


def _is_missing(exc: MlflowException) -> bool:
    return exc.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def fetch_serving_metadata(
    model_uri: str, client: MlflowClient | Any
) -> ServingMetadata:
    """Resolve MODEL_URI once and read threshold, lookup hash and tags.

    Raises:
        ServingStartupError: with a specific message for a missing registered
            model, missing alias or version, missing tags, or a bad threshold.
    """
    name, alias, version = parse_model_uri(model_uri)
    try:
        client.get_registered_model(name)
    except MlflowException as exc:
        if _is_missing(exc):
            raise ServingStartupError(
                f"registered model {name!r} not found; register it with "
                "scripts/register_reference_model.py --apply"
            ) from exc
        raise ServingStartupError(f"cannot read the model registry: {exc}") from exc
    try:
        if alias is not None:
            model_version = client.get_model_version_by_alias(name, alias)
        else:
            model_version = client.get_model_version(name, version)
    except MlflowException as exc:
        if _is_missing(exc):
            what = f"alias {alias!r}" if alias else f"version {version}"
            raise ServingStartupError(
                f"{what} of registered model {name!r} not found"
            ) from exc
        raise ServingStartupError(f"cannot read the model registry: {exc}") from exc
    tags = dict(model_version.tags)
    check_required_tags(tags, f"model {name!r} version {model_version.version}")
    return ServingMetadata(
        model_name=name,
        version=str(model_version.version),
        alias=alias,
        threshold=parse_threshold(tags["tuned_threshold"]),
        lookup_sha256=tags["lookup_sha256"],
        tags=tags,
    )


def configure_mlflow(tracking_uri: str) -> None:
    """Set MLflow's global tracking AND registry URIs from the one config value.

    MLflow internals (for example the registry store's `create_model_version`)
    build default `MlflowClient()` objects that ignore a client's explicit URI and
    read these globals instead. Every entry point must call this before creating
    any client or loading any model (ADR-014).
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)


def load_model(metadata: ServingMetadata, tracking_uri: str) -> Any:
    """Load the exact resolved version (skops trusted types come from MLmodel).

    Raises:
        ServingStartupError: if the model artifacts cannot be loaded.
    """
    configure_mlflow(tracking_uri)
    logger.info("Loading %s", metadata.model_uri)
    try:
        return mlflow.sklearn.load_model(metadata.model_uri)
    except Exception as exc:  # noqa: BLE001 - any load failure blocks startup
        raise ServingStartupError(
            f"cannot load model {metadata.model_uri}: {exc}"
        ) from exc


def find_logged_model_id(
    client: MlflowClient | Any, experiment_name: str, source_run_id: str
) -> str:
    """Return the single LoggedModel named 'model' produced by a training run.

    Raises:
        ValueError: if the experiment or the unique model is not found.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"experiment {experiment_name!r} not found")
    candidates = [
        m
        for m in client.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"source_run_id = '{source_run_id}'",
        )
        if m.name == "model"
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected one 'model' for run {source_run_id}, got {len(candidates)}"
        )
    return str(candidates[0].model_id)
