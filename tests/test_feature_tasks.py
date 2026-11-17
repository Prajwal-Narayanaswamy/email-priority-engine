"""Tests of feature task deduplication before feature engineering (ADR-012)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from src.features.features import FEATURE_COLUMNS, KEEP_COLUMNS
from src.tasks import feature_tasks


@pytest.fixture
def raw_emails(tmp_path: Path) -> Path:
    """Two unique emails from one sender plus a copy with a different UID."""
    frame = pd.DataFrame(
        {
            "uid": ["first", "copy", "second"],
            "from": ["alice@example.com"] * 3,
            "subject": ["First topic", "First topic", "Second topic"],
            "date": ["2001-01-01"] * 3,
            "priority": [0, 0, 1],
            "body": ["unused"] * 3,
        }
    )
    path = tmp_path / "emails.parquet"
    frame.to_parquet(path, index=False)
    return path


def test_feature_task_dedupes_before_sender_frequency(
    raw_emails: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        feature_tasks, "get_run_logger", lambda: logging.getLogger("feature_test")
    )
    output = feature_tasks.feature_engineering_task.fn(
        raw_emails, tmp_path / "features.parquet"
    )
    frame = pd.read_parquet(output)
    assert frame["uid"].tolist() == ["first", "second"]
    assert frame["sender_frequency"].tolist() == [2, 2]
    assert frame["priority"].tolist() == [0, 1]
    assert set(frame.columns) == {*KEEP_COLUMNS, *FEATURE_COLUMNS}
    assert "body" not in frame.columns


def test_feature_task_limit_still_dedupes_selected_rows(
    raw_emails: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        feature_tasks, "get_run_logger", lambda: logging.getLogger("feature_test")
    )
    output = feature_tasks.feature_engineering_task.fn(
        raw_emails, tmp_path / "features.parquet", limit=2
    )
    frame = pd.read_parquet(output)
    assert frame["uid"].tolist() == ["first"]
    assert frame["sender_frequency"].tolist() == [1]
