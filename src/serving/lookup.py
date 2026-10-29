"""Sender-frequency lookup for serving.

`sender_frequency` is a dataset-wide count, so a single request cannot compute
it. The lookup is built once from the feature parquet the model was trained on,
which reproduces exactly the values the model saw.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.features.features import normalize_sender

logger = logging.getLogger(__name__)

#: Lowest value seen in training for any real sender; used for unseen senders.
UNSEEN_SENDER_FREQUENCY: int = 1


def build_sender_lookup(df: pd.DataFrame) -> dict[str, int]:
    """Map normalized sender -> sender_frequency as used in training.

    Args:
        df: Feature frame with raw `from` and `sender_frequency` columns.

    Raises:
        ValueError: if one normalized sender maps to several frequencies.
    """
    senders = df["from"].map(normalize_sender)
    keep = senders != ""
    pairs = pd.DataFrame(
        {"sender": senders[keep], "freq": df.loc[keep, "sender_frequency"]}
    )
    per_sender = pairs.groupby("sender")["freq"].nunique()
    if (per_sender > 1).any():
        raise ValueError("sender_frequency is not constant per normalized sender")
    lookup = pairs.drop_duplicates("sender").set_index("sender")["freq"]
    return {str(k): int(v) for k, v in lookup.items()}


def save_sender_lookup(lookup: dict[str, int], path: Path) -> None:
    """Write the lookup as JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lookup, sort_keys=True), encoding="utf-8")
    logger.info("Wrote %d senders to %s", len(lookup), path)


def load_sender_lookup(path: Path) -> dict[str, int]:
    """Read a lookup written by `save_sender_lookup`."""
    return {k: int(v) for k, v in json.loads(path.read_text("utf-8")).items()}


def sender_frequency_for(
    raw_sender: str | None, lookup: dict[str, int]
) -> tuple[int, bool]:
    """Return (frequency, seen_in_training) for a raw From header.

    Null/empty senders get 0 (as in training); unseen senders get
    `UNSEEN_SENDER_FREQUENCY`.
    """
    sender = normalize_sender(raw_sender)
    if sender == "":
        return 0, False
    if sender in lookup:
        return lookup[sender], True
    return UNSEEN_SENDER_FREQUENCY, False
