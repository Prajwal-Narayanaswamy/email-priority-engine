"""Shared helpers for the serving tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.serving.metadata import REQUIRED_TAGS, ServingMetadata

LOOKUP_SHA = "a" * 64
_MLFLOW_ENV_VARS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_REGISTRY_URI",
    "MLFLOW_ALLOW_FILE_STORE",
)


@contextmanager
def isolated_mlflow() -> Iterator[None]:
    """Snapshot/restore MLflow's global URIs and the env vars that mirror them.

    `mlflow.set_tracking_uri` / `set_registry_uri` also write MLFLOW_TRACKING_URI /
    MLFLOW_REGISTRY_URI into os.environ, so code that calls them must be wrapped.
    The env vars start cleared (MLFLOW_ALLOW_FILE_STORE unset, like a fresh shell).
    """
    import mlflow.tracking._model_registry.utils as registry_utils
    import mlflow.tracking._tracking_service.utils as tracking_utils

    with pytest.MonkeyPatch.context() as patch:
        for name in _MLFLOW_ENV_VARS:
            patch.setenv(name, "")  # records the original so exit restores it
            patch.delenv(name)
        saved = (tracking_utils._tracking_uri, registry_utils._registry_uri)
        try:
            yield
        finally:
            tracking_utils._tracking_uri, registry_utils._registry_uri = saved


def make_synthetic_bundle(root: Path) -> tuple[Path, ServingMetadata]:
    """Write a real (tiny LightGBM) skops model plus lookup as a bundle directory."""
    import skops.io
    import yaml
    from lightgbm import LGBMClassifier
    from sklearn import __version__ as sklearn_version

    from src.features.columns import BEHAVIORAL_FEATURES
    from src.serving.bundle import write_bundle
    from src.serving.lookup import build_header, save_sender_lookup

    rng = np.random.default_rng(0)
    x = pd.DataFrame(
        rng.integers(0, 50, (200, len(BEHAVIORAL_FEATURES))),
        columns=list(BEHAVIORAL_FEATURES),
    )
    y = (x.iloc[:, 0] + rng.integers(0, 30, 200) > 40).astype(int)
    model = LGBMClassifier(n_estimators=5, min_child_samples=5, verbose=-1).fit(x, y)
    parts = root / "parts"
    parts.mkdir(parents=True)
    skops.io.dump(model, parts / "model.skops")
    flavor = {
        "pickled_model": "model.skops",
        "serialization_format": "skops",
        "sklearn_version": sklearn_version,
        "skops_trusted_types": skops.io.get_untrusted_types(file=parts / "model.skops"),
    }
    (parts / "MLmodel").write_text(
        yaml.safe_dump({"flavors": {"sklearn": flavor}}), encoding="utf-8"
    )
    digest = save_sender_lookup(
        {"a@b.com": 3}, parts / "lookup.json", build_header("b" * 64, 10)
    )
    metadata = make_metadata(lookup_sha256=digest)
    bundle_dir = root / "bundle"
    write_bundle(
        bundle_dir,
        metadata,
        parts / "MLmodel",
        parts / "model.skops",
        parts / "lookup.json",
        {"python": "3.12"},
        ["lightgbm==4.7.0"],
    )
    return bundle_dir, metadata


class StubModel:
    """Returns a high score only when sender_is_leadership == 1."""

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        p = frame["sender_is_leadership"].to_numpy(dtype=float) * 0.999
        return np.column_stack([1 - p, p])


def make_tags(**overrides: Any) -> dict[str, str]:
    """Return a complete tag dict for a registered version, with overrides."""
    tags = {
        "tuned_threshold": "0.99",
        "threshold_source_run": "eval-run",
        "lookup_sha256": LOOKUP_SHA,
        "lookup_run_id": "lookup-run",
        "split_fingerprint": "fp",
        "source_run_id": "run",
    }
    assert set(tags) == set(REQUIRED_TAGS)
    tags.update(overrides)
    return {k: v for k, v in tags.items() if v is not None}


def make_metadata(**tag_overrides: Any) -> ServingMetadata:
    """Return ServingMetadata for version 3 of email-priority at @champion."""
    tags = make_tags(**tag_overrides)
    return ServingMetadata(
        model_name="email-priority",
        version="3",
        alias="champion",
        threshold=float(tags["tuned_threshold"]),
        lookup_sha256=tags["lookup_sha256"],
        tags=tags,
    )
