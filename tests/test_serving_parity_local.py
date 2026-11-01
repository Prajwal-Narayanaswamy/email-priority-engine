"""Local-only parity tests against the real data, lookup file and model.

Each test skips cleanly when `data/features/features.parquet`,
`data/reference/sender_frequency.json` or the reference model is unavailable, so
CI (which has none of them) is unaffected.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from src.serving import config
from src.serving.app import app
from src.serving.lookup import LookupFormatError, load_sender_lookup
from src.serving.predictor import Predictor, build_feature_row
from src.serving.registry import (
    ServingMetadata,
    ServingStartupError,
    fetch_serving_metadata,
    load_model,
)
from src.training.train import BEHAVIORAL_FEATURES
from tests.serving_helpers import isolated_mlflow

SEED = 42
REQUEST_COLUMNS = ["from", "subject", "date"]


@pytest.fixture(scope="module")
def real_rows() -> pd.DataFrame:
    """Read only the columns needed from the real parquet, or skip."""
    if not config.FEATURES_PATH.is_file():
        pytest.skip(f"{config.FEATURES_PATH} not available")
    return pd.read_parquet(
        config.FEATURES_PATH, columns=[*REQUEST_COLUMNS, *BEHAVIORAL_FEATURES]
    )


@pytest.fixture(scope="module")
def real_lookup() -> dict[str, int]:
    """Load the real sender lookup file, or skip."""
    if not config.SENDER_LOOKUP_PATH.is_file():
        pytest.skip(f"{config.SENDER_LOOKUP_PATH} not available")
    try:
        return load_sender_lookup(config.SENDER_LOOKUP_PATH)
    except LookupFormatError as exc:
        pytest.skip(
            f"lookup file is stale or malformed ({exc}); rebuild it with "
            "`python -m src.serving.build_lookup`"
        )


@pytest.fixture(scope="module")
def registered() -> Iterator[tuple[Any, ServingMetadata]]:
    """Resolve MODEL_URI and load the model, or skip if not registered yet."""
    with isolated_mlflow():
        try:
            client = MlflowClient(tracking_uri=config.TRACKING_URI)
            metadata = fetch_serving_metadata(config.MODEL_URI, client)
            model = load_model(metadata, config.TRACKING_URI)
        except ServingStartupError as exc:
            pytest.skip(
                f"{config.MODEL_URI} is not registered or not loadable ({exc}); "
                "run scripts.register_reference_model --apply first"
            )
        yield model, metadata


def _none_if_nan(value: Any) -> Any:
    return None if isinstance(value, float) and np.isnan(value) else value


def test_real_rows_features_match_stored_exactly(
    real_rows: pd.DataFrame, real_lookup: dict[str, int]
) -> None:
    sample = real_rows.sample(2000, random_state=SEED)
    mismatches: list[str] = []
    for index, record in sample.to_dict("index").items():
        row, _ = build_feature_row(
            _none_if_nan(record["from"]),
            _none_if_nan(record["subject"]),
            _none_if_nan(record["date"]),
            real_lookup,
        )
        for feature in BEHAVIORAL_FEATURES:
            if row.iloc[0][feature] != record[feature]:
                mismatches.append(
                    f"row {index}: {feature} serving={row.iloc[0][feature]} "
                    f"stored={record[feature]}"
                )
    assert not mismatches, "\n".join(mismatches[:20])


def test_api_score_equals_model_predict_proba(
    real_rows: pd.DataFrame,
    real_lookup: dict[str, int],
    registered: tuple[Any, ServingMetadata],
) -> None:
    reference_model, metadata = registered
    sample = real_rows.sample(200, random_state=SEED)
    expected = reference_model.predict_proba(sample[list(BEHAVIORAL_FEATURES)])[:, 1]
    app.state.predictor = Predictor(reference_model, metadata, real_lookup)
    try:
        with TestClient(app) as client:
            scores = [
                client.post(
                    "/predict",
                    json={
                        "from": _none_if_nan(r["from"]),
                        "subject": _none_if_nan(r["subject"]),
                        "date": _none_if_nan(r["date"]),
                    },
                ).json()["score"]
                for r in sample.to_dict("records")
            ]
    finally:
        app.state.predictor = None
    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-9)
