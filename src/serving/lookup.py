"""Sender-frequency lookup for serving.

`sender_frequency` is a dataset-wide count, so a single request cannot compute
it. The lookup is built once from the feature parquet the model was trained on,
which reproduces exactly the values the model saw.

File format (v2): ``{"header": {...}, "senders": {normalized sender: count}}``.
The header records which parquet the map came from. A file cannot contain its
own hash, so the registry's ``lookup_sha256`` tag is the sha256 of the FILE
bytes, while the header holds the sha256 of the source PARQUET.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.features import normalize_sender

logger = logging.getLogger(__name__)

#: Lowest value seen in training for any real sender; used for unseen senders.
UNSEEN_SENDER_FREQUENCY: int = 1
FORMAT_VERSION: int = 2
BUILDER_VERSION: str = "2.0.0"
HEADER_KEYS: tuple[str, ...] = (
    "format_version",
    "parquet_sha256",
    "row_count",
    "built_at",
    "builder_version",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LookupFormatError(ValueError):
    """Raised when a lookup file has no header or a malformed header."""


@dataclass(frozen=True)
class LookupFile:
    """A loaded lookup: sender map, validated header and the file's sha256."""

    senders: dict[str, int]
    header: dict[str, Any]
    file_sha256: str


def file_sha256(path: Path) -> str:
    """Return the sha256 hex digest of a file's bytes (streamed)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def build_header(parquet_sha256: str, row_count: int) -> dict[str, Any]:
    """Create the v2 header for a lookup built from the given parquet."""
    return {
        "format_version": FORMAT_VERSION,
        "parquet_sha256": parquet_sha256,
        "row_count": int(row_count),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "builder_version": BUILDER_VERSION,
    }


def validate_header(header: object) -> dict[str, Any]:
    """Return the header if well formed, else raise `LookupFormatError`."""
    if not isinstance(header, dict):
        raise LookupFormatError("lookup file has no header; rebuild with build_lookup")
    missing = [key for key in HEADER_KEYS if key not in header]
    if missing:
        raise LookupFormatError(f"lookup header is missing keys: {missing}")
    if header["format_version"] != FORMAT_VERSION:
        raise LookupFormatError(
            f"unsupported lookup format_version {header['format_version']!r}"
        )
    sha = header["parquet_sha256"]
    if not isinstance(sha, str) or not _SHA256_RE.match(sha):
        raise LookupFormatError("lookup header parquet_sha256 is not a sha256 hex")
    rows = header["row_count"]
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise LookupFormatError("lookup header row_count must be a positive integer")
    built_at = header["built_at"]
    try:
        datetime.fromisoformat(built_at)
    except (TypeError, ValueError) as exc:
        raise LookupFormatError("lookup header built_at is not ISO-8601") from exc
    if not isinstance(header["builder_version"], str) or not header["builder_version"]:
        raise LookupFormatError("lookup header builder_version must be a string")
    return header


def save_sender_lookup(
    lookup: dict[str, int], path: Path, header: dict[str, Any]
) -> str:
    """Write the v2 lookup as JSON and return the file's sha256.

    Raises:
        LookupFormatError: if the header is malformed.
    """
    validate_header(header)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"header": header, "senders": lookup}
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    digest = file_sha256(path)
    logger.info("Wrote %d senders to %s (sha256 %s)", len(lookup), path, digest)
    return digest


def load_lookup(path: Path) -> LookupFile:
    """Read and validate a v2 lookup file.

    Raises:
        FileNotFoundError: if the file does not exist.
        LookupFormatError: if the JSON is invalid, has no header or a bad header.
    """
    try:
        payload = json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError as exc:
        raise LookupFormatError(f"lookup file {path} is not valid JSON") from exc
    if not isinstance(payload, dict) or "header" not in payload:
        raise LookupFormatError(
            f"lookup file {path} has no header (v1 format?); rebuild with build_lookup"
        )
    header = validate_header(payload["header"])
    senders = payload.get("senders")
    if not isinstance(senders, dict):
        raise LookupFormatError(f"lookup file {path} has no 'senders' map")
    return LookupFile(
        senders={k: int(v) for k, v in senders.items()},
        header=header,
        file_sha256=file_sha256(path),
    )


def load_sender_lookup(path: Path) -> dict[str, int]:
    """Read a lookup file and return only the sender map (header validated)."""
    return load_lookup(path).senders


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
