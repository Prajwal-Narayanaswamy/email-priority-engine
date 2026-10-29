"""Serving configuration, read from environment variables (python-dotenv)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: ADR-013 serving reference: behavioral-only training run.
DEFAULT_MODEL_RUN_ID: str = "7189707b198f47748f89965712eee4bc"
#: ADR-013 Step 2c evaluation run; holds the validation-selected threshold.
DEFAULT_EVAL_RUN_ID: str = "559ccf6356494eecabb332c47eff2105"

MODEL_RUN_ID: str = os.environ.get("MODEL_RUN_ID", DEFAULT_MODEL_RUN_ID)
EVAL_RUN_ID: str = os.environ.get("EVAL_RUN_ID", DEFAULT_EVAL_RUN_ID)
TRACKING_URI: str = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT}/mlruns/mlflow.db"
)
EXPERIMENT_NAME: str = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME", "email-priority-classifier"
)
SENDER_LOOKUP_PATH: Path = Path(
    os.environ.get(
        "SENDER_LOOKUP_PATH", str(REPO_ROOT / "data/reference/sender_frequency.json")
    )
)
FEATURES_PATH: Path = Path(
    os.environ.get("FEATURES_PATH", str(REPO_ROOT / "data/features/features.parquet"))
)
