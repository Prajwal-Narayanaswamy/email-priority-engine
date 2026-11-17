"""Runtime library versions and the comparison with a bundle's recorded versions.

A mismatch only warns: the model may still load, and refusing to start would turn
a patch-level difference into an outage. The versions are reported by /model-info.
"""

from __future__ import annotations

import logging
import platform
from importlib import metadata as importlib_metadata

logger = logging.getLogger(__name__)

LIBRARIES: tuple[str, ...] = (
    "lightgbm",
    "scikit-learn",
    "numpy",
    "pandas",
    "scipy",
    "skops",
)


def runtime_versions() -> dict[str, str]:
    """Return the Python and model-relevant library versions of this process."""
    versions = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _comparable(name: str, version: str) -> str:
    # Python patch releases are not part of the model's contract.
    return ".".join(version.split(".")[:2]) if name == "python" else version


def version_mismatches(
    runtime: dict[str, str], bundle: dict[str, str]
) -> dict[str, tuple[str, str]]:
    """Return {name: (runtime, bundle)} for every name in both that differs.

    Python is compared by major.minor, everything else exactly.
    """
    return {
        name: (runtime[name], bundle[name])
        for name in LIBRARIES + ("python",)
        if name in runtime
        and name in bundle
        and _comparable(name, runtime[name]) != _comparable(name, bundle[name])
    }


def warn_on_mismatch(runtime: dict[str, str], bundle: dict[str, str] | None) -> bool:
    """Log one WARNING if runtime and bundle versions differ; return whether it did."""
    if not bundle:
        return False
    mismatches = version_mismatches(runtime, bundle)
    if not mismatches:
        return False
    detail = ", ".join(
        f"{name} runtime {have} vs bundle {want}"
        for name, (have, want) in sorted(mismatches.items())
    )
    logger.warning(
        "Runtime library versions differ from those the bundle was exported with "
        "(%s); predictions may differ from the registry model",
        detail,
    )
    return True
