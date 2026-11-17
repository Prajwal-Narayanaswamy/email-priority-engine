"""Serving metadata shared by every model source (registry or bundle).

This module must not import mlflow: the bundle path runs in an image without it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

REQUIRED_TAGS: tuple[str, ...] = (
    "tuned_threshold",
    "threshold_source_run",
    "lookup_sha256",
    "lookup_run_id",
    "split_fingerprint",
    "source_run_id",
)


class ServingStartupError(RuntimeError):
    """Raised at startup when the model source cannot support serving."""


@dataclass(frozen=True)
class ServingMetadata:
    """Everything serving needs to know about the model version it loads."""

    model_name: str
    version: str
    alias: str | None
    threshold: float
    lookup_sha256: str
    tags: dict[str, str]
    #: Library versions the bundle was exported with (None for registry mode).
    bundle_versions: dict[str, str] | None = field(default=None, compare=False)

    @property
    def model_uri(self) -> str:
        """Immutable URI of the resolved version (never the alias)."""
        return f"models:/{self.model_name}/{self.version}"


def parse_threshold(raw: str) -> float:
    """Parse the tuned_threshold tag; it must be finite and inside (0, 1).

    Raises:
        ServingStartupError: if non-numeric or out of range.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ServingStartupError(
            f"tuned_threshold tag {raw!r} is not numeric"
        ) from exc
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ServingStartupError(
            f"tuned_threshold tag {raw!r} is outside the open interval (0, 1)"
        )
    return value


def check_required_tags(tags: dict[str, str], label: str) -> None:
    """Raise ServingStartupError naming every required tag that is missing."""
    missing = [tag for tag in REQUIRED_TAGS if tag not in tags]
    if missing:
        raise ServingStartupError(f"{label} is missing tags: {missing}")
