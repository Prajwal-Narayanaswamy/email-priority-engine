"""Serving configuration, read from environment variables (python-dotenv)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Registry URI of the model to serve (ADR-014); no fallback to run-ID loading.
MODEL_URI: str = os.environ.get("MODEL_URI", "models:/email-priority@champion")
TRACKING_URI: str = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT}/mlruns/mlflow.db"
)
EXPERIMENT_NAME: str = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME", "email-priority-classifier"
)
#: Separate experiment for serving artifacts (never the training experiment).
SERVING_EXPERIMENT_NAME: str = os.environ.get(
    "MLFLOW_SERVING_EXPERIMENT_NAME", "email-priority-serving"
)
SENDER_LOOKUP_PATH: Path = Path(
    os.environ.get(
        "SENDER_LOOKUP_PATH", str(REPO_ROOT / "data/reference/sender_frequency.json")
    )
)
FEATURES_PATH: Path = Path(
    os.environ.get("FEATURES_PATH", str(REPO_ROOT / "data/features/features.parquet"))
)
