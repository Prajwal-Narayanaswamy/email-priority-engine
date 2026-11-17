"""Local-only: bundle loader vs mlflow loader on the real registered model.

Exports the real `MODEL_URI` version to a temporary bundle (read-only against the
registry) and requires identical `predict_proba` from both loaders. Skips when the
registry, model, lookup or features are not available, so CI is unaffected.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from scripts.export_model_bundle import export_bundle
from src.features.columns import BEHAVIORAL_FEATURES
from src.serving import bundle, config
from src.serving.lookup import LookupFormatError
from src.serving.metadata import ServingStartupError
from src.serving.registry import configure_mlflow, load_model
from tests.serving_helpers import isolated_mlflow

FIXTURE = Path(__file__).parent / "fixtures/serving_parity_rows.parquet"
SEED = 42
ATOL = 1e-12


@pytest.fixture(scope="module")
def both_models(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Any, Any]]:
    """(bundle-loaded model, mlflow-loaded model) of the same registered version."""
    out = tmp_path_factory.mktemp("real") / "bundle"
    with isolated_mlflow():
        try:
            configure_mlflow(config.TRACKING_URI)
            client = MlflowClient(tracking_uri=config.TRACKING_URI)
            metadata = export_bundle(
                client, config.MODEL_URI, out, config.SENDER_LOOKUP_PATH, False
            )
            via_mlflow = load_model(metadata, config.TRACKING_URI)
        except (ServingStartupError, FileNotFoundError, LookupFormatError) as exc:
            pytest.skip(f"real registry model or lookup not available ({exc})")
        yield bundle.load_bundle_model(out), via_mlflow


def test_loaders_agree_on_the_300_row_fixture(both_models: tuple[Any, Any]) -> None:
    rows = pd.read_parquet(FIXTURE)[list(BEHAVIORAL_FEATURES)]
    assert len(rows) == 300
    via_bundle, via_mlflow = both_models
    np.testing.assert_allclose(
        via_bundle.predict_proba(rows),
        via_mlflow.predict_proba(rows),
        rtol=0,
        atol=ATOL,
    )


def test_loaders_agree_on_2000_real_rows(both_models: tuple[Any, Any]) -> None:
    if not config.FEATURES_PATH.is_file():
        pytest.skip(f"{config.FEATURES_PATH} not available")
    real = pd.read_parquet(config.FEATURES_PATH, columns=list(BEHAVIORAL_FEATURES))
    rows = real.sample(2000, random_state=SEED)
    via_bundle, via_mlflow = both_models
    np.testing.assert_allclose(
        via_bundle.predict_proba(rows),
        via_mlflow.predict_proba(rows),
        rtol=0,
        atol=ATOL,
    )
