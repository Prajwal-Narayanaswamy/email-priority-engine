"""Runtime-versus-bundle library version comparison."""

from __future__ import annotations

import logging
from importlib import metadata as importlib_metadata

import pytest

from src.serving.versions import (
    LIBRARIES,
    runtime_versions,
    version_mismatches,
    warn_on_mismatch,
)

RUNTIME = {"python": "3.12.13", "lightgbm": "4.7.0", "numpy": "2.5.2"}


def test_runtime_versions_lists_python_and_the_model_libraries() -> None:
    versions = runtime_versions()
    assert set(versions) == {"python", *LIBRARIES}
    assert versions["numpy"] == importlib_metadata.version("numpy")
    assert set(LIBRARIES) == {
        "lightgbm",
        "scikit-learn",
        "numpy",
        "pandas",
        "scipy",
        "skops",
    }


def test_identical_versions_and_python_patch_difference_are_not_mismatches() -> None:
    bundle = {"python": "3.12.3", "lightgbm": "4.7.0", "numpy": "2.5.2"}
    assert version_mismatches(RUNTIME, bundle) == {}


def test_library_and_python_minor_differences_are_reported() -> None:
    bundle = {"python": "3.11.9", "lightgbm": "4.7.1", "numpy": "2.5.2"}
    assert version_mismatches(RUNTIME, bundle) == {
        "python": ("3.12.13", "3.11.9"),
        "lightgbm": ("4.7.0", "4.7.1"),
    }


def test_names_missing_on_either_side_are_ignored() -> None:
    assert version_mismatches({"numpy": "1"}, {"scipy": "2"}) == {}


def test_warn_logs_one_warning_naming_every_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bundle = {"lightgbm": "4.7.1", "numpy": "9.9.9"}
    with caplog.at_level(logging.WARNING, logger="src.serving.versions"):
        assert warn_on_mismatch(RUNTIME, bundle) is True
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "lightgbm runtime 4.7.0 vs bundle 4.7.1" in message
    assert "numpy runtime 2.5.2 vs bundle 9.9.9" in message


@pytest.mark.parametrize("bundle", [None, {}, {"lightgbm": "4.7.0"}])
def test_warn_is_silent_without_differences(
    bundle: dict[str, str] | None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="src.serving.versions"):
        assert warn_on_mismatch(RUNTIME, bundle) is False
    assert caplog.records == []
